from src.components.recovery_engine import recommend, risk_level

RISK = dict(low_delay_minutes=5, medium_probability=0.35, high_delay_minutes=20, critical_delay_minutes=40)


def test_risk_levels():
    assert risk_level(0.10, 30, RISK) == "LOW"        # delay unlikely
    assert risk_level(0.90, 3, RISK) == "LOW"         # delay negligible
    assert risk_level(0.70, 12, RISK) == "MEDIUM"
    assert risk_level(0.90, 25, RISK) == "HIGH"
    assert risk_level(0.99, 55, RISK) == "CRITICAL"


def test_actions_match_risk():
    assert recommend("LOW", [])[0] == "CONTINUE"
    assert recommend("MEDIUM", [])[0] == "MONITOR"
    assert recommend("HIGH", [])[0] == "REROUTE"
    code, actions = recommend("CRITICAL", [])
    assert code == "PRIORITIZE_AND_NOTIFY" and any("Notify" in a for a in actions)


def test_rain_tip_only_for_two_wheelers():
    f = [{"factor": "Rain", "impact_minutes": 8, "contribution_pct": 50}]
    assert any("switching" in a for a in recommend("HIGH", f, "bike")[1])
    assert not any("switching" in a for a in recommend("HIGH", f, "car")[1])
