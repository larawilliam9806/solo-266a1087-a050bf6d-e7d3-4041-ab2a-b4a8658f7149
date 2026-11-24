"""端到端 API 测试：覆盖登记、判定、发布、暂停、恢复的完整生命周期。"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def register_training(client, scan_id):
    r = client.post("/training-scans", json={"scan_id": scan_id})
    assert r.status_code in (200, 201), r.text
    return r


def link(client, evidence_id, a, b):
    r = client.post(
        "/same-page-links",
        json={"evidence_id": evidence_id, "scan_a": a, "scan_b": b},
    )
    assert r.status_code in (200, 201), r.text
    return r


def unlink(client, evidence_id):
    r = client.delete(f"/same-page-links/{evidence_id}")
    assert r.status_code == 200, r.text
    return r


def publish(client, items):
    r = client.post("/manifests", json={"items": items})
    assert r.status_code == 201, r.text
    return r.json()["manifest_id"]


# ---------------------------------------------------------------------------
# 清单提交：退回与发布
# ---------------------------------------------------------------------------

def test_publish_clean_manifest(client):
    r = client.post("/manifests", json={"items": ["S1", "S2", "S2", "S3"]})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "published"
    assert body["items"] == ["S1", "S2", "S3"]  # 去重保序
    mid = body["manifest_id"]

    r = client.get(f"/manifests/{mid}")
    assert r.status_code == 200
    assert r.json() == {"manifest_id": mid, "status": "published",
                        "items": ["S1", "S2", "S3"]}


def test_reject_manifest_containing_training_scan(client):
    register_training(client, "T1")
    r = client.post("/manifests", json={"items": ["S1", "T1"]})
    assert r.status_code == 409
    assert r.json()["conflicts"] == [{"eval_scan_id": "T1", "training_scan_id": "T1"}]
    assert "manifest_id" not in r.json()


def test_reject_manifest_indirect_same_page(client):
    # T1 -A- B：B 与训练件 T1 间接相连，也算同页
    register_training(client, "T1")
    link(client, "E1", "T1", "A")
    link(client, "E2", "A", "B")
    r = client.post("/manifests", json={"items": ["B", "S9"]})
    assert r.status_code == 409
    assert r.json()["conflicts"] == [{"eval_scan_id": "B", "training_scan_id": "T1"}]


def test_failed_submission_leaves_state_unchanged(client):
    register_training(client, "T1")
    assert client.post("/manifests", json={"items": ["T1"]}).status_code == 409
    client.delete("/training-scans/T1")
    # 冲突源消除后同一批件号可以发布，说明退回时未写入任何残留状态
    r = client.post("/manifests", json={"items": ["T1"]})
    assert r.status_code == 201


# ---------------------------------------------------------------------------
# 已发布清单：出现冲突即暂停，消除后按原编号恢复
# ---------------------------------------------------------------------------

def test_suspend_on_new_same_page_link_then_recover(client):
    mid = publish(client, ["A", "S1"])
    register_training(client, "T1")

    # 尚无关联，仍供出原清单
    assert client.get(f"/manifests/{mid}").json()["status"] == "published"

    # 新判定 A 与 T1 同页 -> 暂停供数，只查到冲突对
    link(client, "E1", "A", "T1")
    r = client.get(f"/manifests/{mid}")
    assert r.json() == {
        "manifest_id": mid,
        "status": "suspended",
        "conflicts": [{"eval_scan_id": "A", "training_scan_id": "T1"}],
    }
    assert "items" not in r.json()

    # 撤回判定 -> 冲突消除，按原编号供出原清单
    unlink(client, "E1")
    r = client.get(f"/manifests/{mid}")
    assert r.json() == {"manifest_id": mid, "status": "published",
                        "items": ["A", "S1"]}


def test_suspend_on_new_training_registration(client):
    link(client, "E1", "A", "B")
    mid = publish(client, ["A"])
    assert client.get(f"/manifests/{mid}").json()["status"] == "published"

    # 新登记的训练件 B 与 A 同页 -> 暂停
    register_training(client, "B")
    r = client.get(f"/manifests/{mid}")
    assert r.json()["status"] == "suspended"
    assert r.json()["conflicts"] == [{"eval_scan_id": "A", "training_scan_id": "B"}]

    # 注销训练件 -> 恢复
    client.delete("/training-scans/B")
    assert client.get(f"/manifests/{mid}").json()["status"] == "published"


def test_transitive_conflict_and_partial_withdrawal(client):
    # X -Y- Z，训练件在 Z，清单含 X -> 间接同页冲突
    link(client, "E1", "X", "Y")
    link(client, "E2", "Y", "Z")
    mid = publish(client, ["X"])
    register_training(client, "Z")
    assert client.get(f"/manifests/{mid}").json()["conflicts"] == [
        {"eval_scan_id": "X", "training_scan_id": "Z"}
    ]

    # 撤回 E2 后 X/Y 与 Z 断开 -> 恢复
    unlink(client, "E2")
    assert client.get(f"/manifests/{mid}").json()["status"] == "published"


def test_withdrawal_without_split_keeps_suspension(client):
    # 环：A-B-C-A，撤回一条边分量不分裂，冲突仍在
    link(client, "E1", "A", "B")
    link(client, "E2", "B", "C")
    link(client, "E3", "A", "C")
    mid = publish(client, ["A"])
    register_training(client, "C")
    assert client.get(f"/manifests/{mid}").json()["status"] == "suspended"

    unlink(client, "E3")  # 仍有路径 A-B-C
    assert client.get(f"/manifests/{mid}").json()["status"] == "suspended"

    unlink(client, "E2")  # 彻底断开
    assert client.get(f"/manifests/{mid}").json()["status"] == "published"


def test_direct_training_conflict_resolved_by_deregister(client):
    mid = publish(client, ["A"])
    register_training(client, "A")
    r = client.get(f"/manifests/{mid}")
    assert r.json()["conflicts"] == [{"eval_scan_id": "A", "training_scan_id": "A"}]

    client.delete("/training-scans/A")
    r = client.get(f"/manifests/{mid}")
    assert r.json() == {"manifest_id": mid, "status": "published", "items": ["A"]}


def test_multiple_manifests_and_multiple_conflicts(client):
    m1 = publish(client, ["A"])
    m2 = publish(client, ["B", "S1"])
    link(client, "E1", "A", "B")
    register_training(client, "B")

    # 分量 {A, B} 含训练件 B：m1 的 A、m2 的 B 都冲突
    assert client.get(f"/manifests/{m1}").json()["conflicts"] == [
        {"eval_scan_id": "A", "training_scan_id": "B"}
    ]
    assert client.get(f"/manifests/{m2}").json()["conflicts"] == [
        {"eval_scan_id": "B", "training_scan_id": "B"}
    ]

    unlink(client, "E1")
    assert client.get(f"/manifests/{m1}").json()["status"] == "published"
    # m2 的 B 本身是训练件，冲突仍在
    assert client.get(f"/manifests/{m2}").json()["status"] == "suspended"


# ---------------------------------------------------------------------------
# 大分量撤回：不依赖分量成员数量展开 SQL 参数
# ---------------------------------------------------------------------------

def _seed_chain(client, n):
    """T0 -E1- T1 -E2- ... -En- Tn，全部落在同一分量。"""
    for i in range(1, n + 1):
        link(client, f"E{i}", f"T{i - 1}", f"T{i}")


def test_withdraw_from_large_group_splits_at_faulty_link(client):
    """大量扫描件被误判串成一组时，撤回中间判定只拆该组，清单按原编号恢复。"""
    _seed_chain(client, 120)  # 121 个成员的同页组
    link(client, "E-close", "T0", "T120")  # 成环：撤回一条判定后仍有其他路径

    m0 = publish(client, ["T0", "S1"])
    m120 = publish(client, ["T120"])
    register_training(client, "T120")  # 训练件在组末端

    # T0 与训练件间接同页 -> 暂停；T120 自身即训练件
    assert client.get(f"/manifests/{m0}").json()["status"] == "suspended"
    assert client.get(f"/manifests/{m120}").json()["status"] == "suspended"

    # 撤回中间判定：仍有 T0-T120 环路，分量不分裂，冲突明细照旧
    assert client.delete("/same-page-links/E60").status_code == 200
    assert client.get(f"/manifests/{m0}").json()["conflicts"] == [
        {"eval_scan_id": "T0", "training_scan_id": "T120"}
    ]

    # 再撤回环路补边：T0 侧与 T120 侧断开
    assert client.delete("/same-page-links/E-close").status_code == 200
    # T0 侧冲突清空，按原编号和顺序恢复供数
    assert client.get(f"/manifests/{m0}").json() == {
        "manifest_id": m0, "status": "published", "items": ["T0", "S1"]
    }
    # T120 侧仍含训练件，只保留仍然同页的冲突
    assert client.get(f"/manifests/{m120}").json()["conflicts"] == [
        {"eval_scan_id": "T120", "training_scan_id": "T120"}
    ]


def test_withdrawal_does_not_touch_unrelated_groups(client):
    """拆组核对只读取原组内的有效判定，另一个同页组的冲突不受影响。"""
    link(client, "E-own", "A", "B")
    link(client, "E-other", "X", "Y")
    m_own = publish(client, ["A"])
    m_other = publish(client, ["X"])
    register_training(client, "B")
    register_training(client, "Y")

    assert client.get(f"/manifests/{m_own}").json()["status"] == "suspended"
    assert client.get(f"/manifests/{m_other}").json()["status"] == "suspended"

    # 撤回无关组内的判定（X-Y 拆开），不应波及 A/B 组
    assert client.delete("/same-page-links/E-other").status_code == 200
    assert client.get(f"/manifests/{m_other}").json()["status"] == "published"
    assert client.get(f"/manifests/{m_own}").json()["conflicts"] == [
        {"eval_scan_id": "A", "training_scan_id": "B"}
    ]


def test_withdraw_large_group_under_low_param_limit(tmp_path):
    """把连接参数上限压到 100，撤回 121 成员分量仍成功（回归 500）。"""
    import sqlite3

    import app.db as appdb

    db_path = str(tmp_path / "test.db")
    original_connect = appdb.connect

    def limited_connect(path):
        conn = original_connect(path)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 100)
        return conn

    appdb.connect = limited_connect
    try:
        app = create_app(db_path)
        with TestClient(app) as c:
            _seed_chain(c, 120)  # 121 成员：旧的 IN 列表需要 242 个参数
            mid = publish(c, ["T0"])
            register_training(c, "T120")
            assert c.get(f"/manifests/{mid}").json()["status"] == "suspended"

            r = c.delete("/same-page-links/E60")
            assert r.status_code == 200, r.text
            assert c.get(f"/manifests/{mid}").json() == {
                "manifest_id": mid, "status": "published", "items": ["T0"]
            }
    finally:
        appdb.connect = original_connect


def test_withdraw_failure_rolls_back_to_pre_withdrawal_state(tmp_path, monkeypatch):
    """拆组/供数写入失败时整体回滚：判定仍有效，清单仍停供。"""
    import app.store as appstore

    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app, raise_server_exceptions=False) as c:
        link(c, "E1", "A", "T1")
        mid = publish(c, ["A"])
        register_training(c, "T1")
        assert c.get(f"/manifests/{mid}").json()["status"] == "suspended"

        def boom(*_args, **_kwargs):
            raise RuntimeError("write failed")

        monkeypatch.setattr(appstore, "_recompute_conflicts", boom)
        r = c.delete("/same-page-links/E1")
        assert r.status_code == 500
        monkeypatch.undo()

        links = c.get("/same-page-links").json()["items"]
        assert [(x["evidence_id"], x["active"]) for x in links] == [("E1", 1)]
        # 冲突与供数结论也保持撤回前状态
        assert c.get(f"/manifests/{mid}").json() == {
            "manifest_id": mid,
            "status": "suspended",
            "conflicts": [{"eval_scan_id": "A", "training_scan_id": "T1"}],
        }


# ---------------------------------------------------------------------------
# 登记与判定的边界行为
# ---------------------------------------------------------------------------

def test_training_registration_idempotent(client):
    register_training(client, "T1")
    r = register_training(client, "T1")
    assert r.status_code == 200
    assert r.json()["created"] is False
    assert client.get("/training-scans").json()["items"] == ["T1"]


def test_deregister_unknown_training_scan_404(client):
    assert client.delete("/training-scans/NOPE").status_code == 404


def test_evidence_id_reuse_with_different_pair_rejected(client):
    link(client, "E1", "A", "B")
    r = client.post(
        "/same-page-links",
        json={"evidence_id": "E1", "scan_a": "A", "scan_b": "C"},
    )
    assert r.status_code == 409


def test_same_page_link_idempotent_and_reactivation(client):
    link(client, "E1", "A", "B")
    r = link(client, "E1", "A", "B")
    assert r.status_code == 200 and r.json()["created"] is False

    unlink(client, "E1")
    unlink(client, "E1")  # 重复撤回幂等
    r = link(client, "E1", "B", "A")  # 同一对（顺序无关）重新登记
    assert r.status_code == 201 and r.json()["created"] is True

    links = client.get("/same-page-links").json()["items"]
    assert len(links) == 1 and links[0]["active"] == 1


def test_withdraw_unknown_evidence_404(client):
    assert client.delete("/same-page-links/NOPE").status_code == 404


def test_same_scan_pair_rejected(client):
    r = client.post(
        "/same-page-links",
        json={"evidence_id": "E1", "scan_a": "A", "scan_b": "A"},
    )
    assert r.status_code == 422


def test_unknown_manifest_404(client):
    assert client.get("/manifests/MFT-UNKNOWN").status_code == 404


def test_empty_or_blank_input_rejected(client):
    assert client.post("/manifests", json={"items": []}).status_code == 422
    assert client.post("/manifests", json={"items": ["  "]}).status_code == 422
    assert client.post("/training-scans", json={"scan_id": " "}).status_code == 422
