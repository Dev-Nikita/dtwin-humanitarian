"""The post-confirmation controls must not change any confirmed policy."""

from python.orchestrator.policies import POLICY_SPECS, REFRESH_COOLDOWN


def test_confirmed_policies_keep_refresh_cooldown():
    for name, spec in POLICY_SPECS.items():
        if name != "protection_predictive_slow":
            assert spec.refresh_cooldown == REFRESH_COOLDOWN, name


def test_repair_control_uses_no_risk():
    spec = POLICY_SPECS["protection_repair"]
    assert spec.risk == "none" and spec.restoration == "protection"
    assert spec.backup_refresh and not spec.migrate


def test_slow_variant_differs_from_p6_only_in_cooldown():
    p6, slow = POLICY_SPECS["protection_predictive"], POLICY_SPECS["protection_predictive_slow"]
    assert slow.refresh_cooldown == 5 * REFRESH_COOLDOWN
    assert (p6.risk, p6.restoration, p6.migrate, p6.backup_refresh, p6.primary_risk) == \
        (slow.risk, slow.restoration, slow.migrate, slow.backup_refresh, slow.primary_risk)
