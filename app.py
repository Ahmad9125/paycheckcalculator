import json
import os
import re
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

try:
    import pytesseract
except ImportError:  # Friendly error if dependency is missing
    pytesseract = None

CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_default_settings(cfg: dict) -> dict:
    defaults = {
        "rate_overrides": {
        "regular": float(cfg.get("base_hourly_rate", cfg.get("regular_hourly_rate", 0.0))),
        "pto": float(cfg.get("pto_hourly_rate", cfg.get("base_hourly_rate", 0.0))),
        "overtime": float(cfg["overtime_hourly_rate"]),
        "holiday": float(cfg["holiday_hourly_rate"]),
        "evening_diff": float(cfg.get("evening_diff_per_hour", cfg.get("shift_diff_per_hour", 0.0))),
        "weekend_diff": float(cfg.get("weekend_diff_per_hour", cfg.get("shift_diff_per_hour", 0.0))),
        "night_diff": float(cfg.get("night_diff_per_hour", cfg.get("shift_diff_per_hour", 0.0))),
    },
        "pto_policy": {
            "years_of_service": 0.0,
            "annual_pto_override": 0.0,
            "pay_periods": 26,
            "eligible_hours_cap": 80.0,
        },
        "benefits": {
            "k401_percent": float(cfg.get("k401_percent", 0.0)),
            "dental_pretax": float(cfg.get("pretax_benefits_per_paycheck", {}).get("dental_pretax", 0.0)),
            "health_pretax": float(cfg.get("pretax_benefits_per_paycheck", {}).get("health_pretax", 0.0)),
            "vision_pretax": float(cfg.get("pretax_benefits_per_paycheck", {}).get("vision_pretax", 0.0)),
        },
        "tax": {
            "state_label": "IL State (est.)",
        },
    }
    return defaults


def ocr_extract_hours(file_bytes: bytes):
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Run `pip install pytesseract pillow` and install Tesseract OCR.")

    img = Image.open(BytesIO(file_bytes))
    text = pytesseract.image_to_string(img)

    def last_number(line: str):
        nums = re.findall(r"[-+]?[0-9]+(?:\.[0-9]+)?", line)
        return float(nums[-1]) if nums else None

    # initialize
    extracted = {
        "regular": 0.0,
        "overtime": 0.0,
        "pto": 0.0,
        "holiday": 0.0,
        "evening_diff": 0.0,
        "weekend_diff": 0.0,
        "night_diff": 0.0,
    }
    found = {k: False for k in extracted.keys()}

    # Line-based heuristics for common timecard labels (REG∑, UPTO∑, WKND, etc.)
    for line in text.splitlines():
        l = line.lower()
        if "unpd" in l or "unpaid" in l:
            continue  # ignore unpaid lines
        if "total worked" in l:
            continue  # skip grand totals row
        val = last_number(line)
        if val is None:
            continue
        if re.search(r"\breg", l):
            extracted["regular"] = val
            found["regular"] = True
        elif "pto" in l or "upto" in l:
            extracted["pto"] = val
            found["pto"] = True
        elif "wknd" in l or "weekend" in l:
            extracted["weekend_diff"] = val
            found["weekend_diff"] = True
        elif "hol" in l:
            extracted["holiday"] = val
            found["holiday"] = True
        elif re.search(r"\bot\b", l) and "tot" not in l:  # only explicit OT labels
            extracted["overtime"] = val
            found["overtime"] = True
        elif "eve" in l or "evening" in l:
            extracted["evening_diff"] = val
            found["evening_diff"] = True
        elif "night" in l:
            extracted["night_diff"] = val
            found["night_diff"] = True

    # Regex fallback on text stripped of "total" lines to avoid OT mis-matches
    filtered_text = "\n".join([line for line in text.splitlines() if "tot" not in line.lower()])
    patterns = {
        "regular": r"(?:reg|rqg|regular)[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        "overtime": r"(?:^|\\s)(?:ot|overtime)[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        "pto": r"(?:pto|upto)[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        "holiday": r"holiday[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        "evening_diff": r"evening\\s*diff[^0-9]*([0-9]+(?:\\.[0-9]+)?)",
        "weekend_diff": r"(?:weekend|wknd)[^0-9]*([0-9]+(?:\\.[0-9]+)?)",
        "night_diff": r"night[^0-9]*([0-9]+(?:\\.[0-9]+)?)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, filtered_text, flags=re.IGNORECASE | re.MULTILINE)
        if match and extracted[key] == 0.0:
            extracted[key] = float(match.group(1))
            found[key] = True

    # Totals block fallback: after a line containing 'Totals', map following decimal numbers
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "totals" in line.lower():
            nums = []
            for following in lines[idx + 1 : idx + 10]:
                val = last_number(following)
                if val is not None and "." in following:
                    nums.append(val)
            if len(nums) >= 1 and not found["regular"]:
                extracted["regular"] = nums[0]
                found["regular"] = True
            if len(nums) >= 2 and not found["weekend_diff"]:
                extracted["weekend_diff"] = nums[1]
                found["weekend_diff"] = True
            break

    # Heuristic: if REG missing but we captured a single other positive number, treat it as REG
    if not found["regular"]:
        candidates = [
            extracted[k]
            for k in ["overtime", "pto", "holiday", "weekend_diff", "evening_diff", "night_diff"]
            if extracted[k] > 0
        ]
        if len(candidates) == 1:
            extracted["regular"] = candidates[0]
            for k in ["overtime", "pto", "holiday", "weekend_diff", "evening_diff", "night_diff"]:
                if extracted[k] == extracted["regular"]:
                    extracted[k] = 0.0
            found["regular"] = True

    return text, extracted


def compute(cfg, totals):
    base = float(cfg.get("base_hourly_rate", cfg.get("regular_hourly_rate", 0.0)))
    pto_rate = float(cfg.get("pto_hourly_rate", base))
    ot_rate = float(cfg["overtime_hourly_rate"])
    hol_rate = float(cfg["holiday_hourly_rate"])
    eve_diff = float(cfg.get("evening_diff_per_hour", cfg.get("shift_diff_per_hour", 0.0)))
    wknd_diff = float(cfg.get("weekend_diff_per_hour", cfg.get("shift_diff_per_hour", 0.0)))
    night_diff = float(cfg.get("night_diff_per_hour", cfg.get("shift_diff_per_hour", 0.0)))

    # Earnings
    regular_pay = totals["regular"] * base
    pto_pay = totals["pto"] * pto_rate
    holiday_pay = totals["holiday"] * hol_rate
    overtime_pay = totals["overtime"] * ot_rate
    diff_pay = (
        (totals["evening_diff"] * eve_diff)
        + (totals["weekend_diff"] * wknd_diff)
        + (totals["night_diff"] * night_diff)
    )

    gross = regular_pay + pto_pay + holiday_pay + overtime_pay + diff_pay

    # Pre-tax deductions
    k401 = gross * float(cfg["k401_percent"])

    b = cfg.get("pretax_benefits_per_paycheck", {})
    dental = float(b.get("dental_pretax", 0.0))
    health = float(b.get("health_pretax", 0.0))
    vision = float(b.get("vision_pretax", 0.0))

    cafeteria = dental + health + vision
    pretax_total = k401 + cafeteria

    # Taxable wages
    # 401(k) does NOT reduce FICA wages; cafeteria pre-tax benefits DO reduce FICA wages.
    fica_wages = max(gross - cafeteria, 0.0)
    income_wages = max(gross - pretax_total, 0.0)

    # Taxes
    oasdi = fica_wages * float(cfg["fica_rates"]["oasdi"])
    medicare = fica_wages * float(cfg["fica_rates"]["medicare"])

    wh = cfg["withholding_estimates"]
    fed = income_wages * float(wh["federal_effective_rate"])
    state = income_wages * float(wh["state_il_effective_rate"])
    state_label = cfg.get("state_label", "State (est.)")

    taxes_total = oasdi + medicare + fed + state
    net = gross - pretax_total - taxes_total

    return {
        "gross": gross,
        "net": net,
        "earnings": {
            "Regular": regular_pay,
            "PTO": pto_pay,
            "Holiday": holiday_pay,
            "Overtime": overtime_pay,
            "Shift diff": diff_pay,
            "GROSS": gross,
        },
        "pretax": {
            "401(k)": k401,
            "Dental": dental,
            "Health": health,
            "Vision": vision,
            "Pre-tax total": pretax_total,
        },
        "wages": {
            "FICA taxable wages": fica_wages,
            "Fed/State taxable wages": income_wages,
        },
        "taxes": {
            "OASDI": oasdi,
            "Medicare": medicare,
            "Federal (est.)": fed,
            state_label: state,
            "Taxes total": taxes_total,
        },
    }


def init_session_defaults(cfg: dict, user_settings: dict):
    if "rate_overrides" not in st.session_state:
        st.session_state["rate_overrides"] = user_settings["rate_overrides"]
    if "pto_policy" not in st.session_state:
        st.session_state["pto_policy"] = user_settings["pto_policy"]
    if "benefits" not in st.session_state:
        st.session_state["benefits"] = user_settings["benefits"]
    if "tax" not in st.session_state:
        st.session_state["tax"] = user_settings["tax"]


def apply_overrides(cfg: dict) -> dict:
    cfg = dict(cfg)
    rates = st.session_state["rate_overrides"]
    benefits = st.session_state["benefits"]
    tax = st.session_state["tax"]
    cfg["base_hourly_rate"] = rates["regular"]
    cfg["regular_hourly_rate"] = rates["regular"]
    cfg["pto_hourly_rate"] = rates["pto"]
    cfg["overtime_hourly_rate"] = rates["overtime"]
    cfg["holiday_hourly_rate"] = rates["holiday"]
    cfg["evening_diff_per_hour"] = rates["evening_diff"]
    cfg["weekend_diff_per_hour"] = rates["weekend_diff"]
    cfg["night_diff_per_hour"] = rates["night_diff"]
    cfg["k401_percent"] = float(benefits["k401_percent"])
    cfg["pretax_benefits_per_paycheck"] = {
        "dental_pretax": float(benefits["dental_pretax"]),
        "health_pretax": float(benefits["health_pretax"]),
        "vision_pretax": float(benefits["vision_pretax"]),
    }
    cfg["state_label"] = tax.get("state_label", "State (est.)")
    return cfg


PTO_ACCRUAL_TABLE = [
    {"min_years": 0, "max_years": 3, "rate_per_hour": 0.0962, "per_pay_max": 7.69, "annual_max": 200},
    {"min_years": 3, "max_years": 5, "rate_per_hour": 0.1038, "per_pay_max": 8.31, "annual_max": 216},
    {"min_years": 5, "max_years": 10, "rate_per_hour": 0.1115, "per_pay_max": 8.92, "annual_max": 232},
    {"min_years": 10, "max_years": 15, "rate_per_hour": 0.1231, "per_pay_max": 9.85, "annual_max": 256},
    {"min_years": 15, "max_years": 99, "rate_per_hour": 0.1308, "per_pay_max": 10.46, "annual_max": 272},
]


def pto_bracket(years_of_service: float):
    for row in PTO_ACCRUAL_TABLE:
        if row["min_years"] <= years_of_service < row["max_years"]:
            return row
    return PTO_ACCRUAL_TABLE[-1]


def compute_pto_accrual(years_of_service: float, eligible_hours: float, pay_periods: int, annual_override=0.0):
    if pay_periods <= 0:
        pay_periods = 26

    bracket = pto_bracket(years_of_service)
    rate = bracket["rate_per_hour"]
    per_pay_cap = bracket["per_pay_max"]
    annual_cap = bracket["annual_max"]

    if annual_override and annual_override > 0:
        annual_cap = annual_override
        per_pay_cap = annual_cap / float(pay_periods)

    earned = eligible_hours * rate
    accrued_this_check = min(earned, per_pay_cap)

    projected_annual = min(annual_cap, accrued_this_check * pay_periods)
    return {
        "accrued": accrued_this_check,
        "eligible_hours_used": eligible_hours,
        "rate_per_hour": rate,
        "per_pay_cap": per_pay_cap,
        "annual_cap": annual_cap,
        "projected_annual": projected_annual,
    }


def estimate_pto_accrual(years_of_service: float, pay_periods: int, annual_override=None):
    """Legacy placeholder (unused after PTO table addition)."""
    if pay_periods <= 0:
        pay_periods = 26
    annual_pto = annual_override or 0.0
    accrued_this_check = annual_pto / float(pay_periods) if annual_pto else 0.0
    return annual_pto, accrued_this_check


def week_inputs(week_label: str, key_prefix: str):
    st.markdown(f"### {week_label}")
    c1, c2 = st.columns(2)

    with c1:
        regular = st.number_input(
            f"{week_label} - Regular hours",
            min_value=0.0, step=0.25, value=st.session_state.get(f"{key_prefix}_regular", 0.0), key=f"{key_prefix}_regular"
        )
        overtime = st.number_input(
            f"{week_label} - Overtime hours",
            min_value=0.0, step=0.25, value=st.session_state.get(f"{key_prefix}_overtime", 0.0), key=f"{key_prefix}_overtime"
        )
        pto = st.number_input(
            f"{week_label} - PTO hours",
            min_value=0.0, step=0.25, value=st.session_state.get(f"{key_prefix}_pto", 0.0), key=f"{key_prefix}_pto"
        )

    with c2:
        holiday = st.number_input(
            f"{week_label} - Holiday premium hours",
            min_value=0.0, step=0.25, value=st.session_state.get(f"{key_prefix}_holiday", 0.0), key=f"{key_prefix}_holiday"
        )
        evening_diff = st.number_input(
            f"{week_label} - Evening diff hours",
            min_value=0.0, step=0.25, value=st.session_state.get(f"{key_prefix}_evening", 0.0), key=f"{key_prefix}_evening"
        )
        weekend_diff = st.number_input(
            f"{week_label} - Weekend diff hours (Sat + Sun)",
            min_value=0.0, step=0.25, value=st.session_state.get(f"{key_prefix}_weekend", 0.0), key=f"{key_prefix}_weekend"
        )
        night_diff = st.number_input(
            f"{week_label} - Night diff hours",
            min_value=0.0, step=0.25, value=st.session_state.get(f"{key_prefix}_night", 0.0), key=f"{key_prefix}_night"
        )

    return {
        "regular": regular,
        "overtime": overtime,
        "holiday": holiday,
        "pto": pto,
        "evening_diff": evening_diff,
        "weekend_diff": weekend_diff,
        "night_diff": night_diff,
    }


st.set_page_config(page_title="Bi-Weekly Paycheck Estimator", layout="centered")
cfg = load_config()
default_settings = load_default_settings(cfg)
init_session_defaults(cfg, default_settings)
tabs = st.tabs(["Calculator", "Settings"])

with tabs[1]:
    st.subheader("Rates & PTO settings")
    rates = dict(st.session_state["rate_overrides"])
    pto_policy = dict(st.session_state["pto_policy"])
    pto_policy.setdefault("eligible_hours_cap", 80.0)
    benefits = dict(st.session_state["benefits"])
    tax = dict(st.session_state["tax"])
    with st.form("settings_form"):
        st.markdown("**Hourly rates**")
        r1, r2, r3 = st.columns(3)
        with r1:
            rates["regular"] = st.number_input("Regular hourly rate", min_value=0.0, value=rates["regular"], step=0.25)
            rates["holiday"] = st.number_input("Holiday premium rate", min_value=0.0, value=rates["holiday"], step=0.25)
        with r2:
            rates["overtime"] = st.number_input("Overtime hourly rate", min_value=0.0, value=rates["overtime"], step=0.25)
            rates["pto"] = st.number_input("PTO hourly rate", min_value=0.0, value=rates["pto"], step=0.25)
        with r3:
            rates["evening_diff"] = st.number_input("Evening diff per hour", min_value=0.0, value=rates["evening_diff"], step=0.10, format="%.2f")
            rates["weekend_diff"] = st.number_input("Weekend diff per hour", min_value=0.0, value=rates["weekend_diff"], step=0.10, format="%.2f")
            rates["night_diff"] = st.number_input("Night diff per hour", min_value=0.0, value=rates["night_diff"], step=0.10, format="%.2f")

        st.markdown("**PTO accrual**")
        pto_cols = st.columns(4)
        with pto_cols[0]:
            pto_policy["years_of_service"] = st.number_input("Years of service", min_value=0.0, value=float(pto_policy["years_of_service"]), step=0.5)
        with pto_cols[1]:
            pto_policy["annual_pto_override"] = st.number_input("Annual PTO hours (0 = use default tier)", min_value=0.0, value=float(pto_policy["annual_pto_override"]), step=1.0)
        with pto_cols[2]:
            pto_policy["pay_periods"] = st.number_input("Pay periods per year", min_value=1, max_value=52, value=int(pto_policy["pay_periods"]), step=1)
        with pto_cols[3]:
            pto_policy["eligible_hours_cap"] = st.number_input("Eligible hours cap per pay", min_value=0.0, value=float(pto_policy["eligible_hours_cap"]), step=1.0, format="%.0f", help="Hours above this do not accrue PTO (policy default: 80)")

        st.markdown("**401(k) and pre-tax premiums**")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            benefits["k401_percent"] = st.number_input("401(k) contribution %", min_value=0.0, max_value=1.0, value=float(benefits["k401_percent"]), step=0.01, format="%.2f")
        with b2:
            benefits["dental_pretax"] = st.number_input("Dental premium per paycheck", min_value=0.0, value=float(benefits["dental_pretax"]), step=1.0, format="%.2f")
        with b3:
            benefits["health_pretax"] = st.number_input("Health premium per paycheck", min_value=0.0, value=float(benefits["health_pretax"]), step=1.0, format="%.2f")
        with b4:
            benefits["vision_pretax"] = st.number_input("Vision premium per paycheck", min_value=0.0, value=float(benefits["vision_pretax"]), step=0.50, format="%.2f")

        st.markdown("**Tax display**")
        tax["state_label"] = st.text_input("State line label", value=tax.get("state_label", "State (est.)"), help="Shown in the Taxes breakdown")

        submitted = st.form_submit_button("Save settings")
        if submitted:
            st.session_state["rate_overrides"] = rates
            st.session_state["pto_policy"] = pto_policy
            st.session_state["benefits"] = benefits
            st.session_state["tax"] = tax
            st.success("Settings saved for this browser session.")

with tabs[0]:
    st.title("Bi-Weekly Paycheck Estimator")
    st.caption("Enter Week 1 and Week 2 hours. The app totals them automatically and estimates your net pay.")

    DEFAULT_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if pytesseract is None:
        st.warning("OCR needs `pytesseract` + `Pillow` and the Tesseract engine installed locally.")
    else:
        if DEFAULT_TESSERACT.exists():
            pytesseract.pytesseract.tesseract_cmd = str(DEFAULT_TESSERACT)
            st.info(f"Using Tesseract at {DEFAULT_TESSERACT}")
        else:
            tess_cmd = st.text_input(
                "Tesseract executable path (set if not on PATH)",
                value=os.environ.get("TESSERACT_CMD", ""),
                placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            )
            if tess_cmd.strip():
                pytesseract.pytesseract.tesseract_cmd = tess_cmd.strip()

    st.subheader("Upload a screenshot (single week) to auto-fill hours")
    uploaded = st.file_uploader("Upload PNG/JPG screenshot of your weekly hours", type=["png", "jpg", "jpeg"])
    ocr_text = None
    ocr_hours = None
    if uploaded:
        try:
            ocr_text, ocr_hours = ocr_extract_hours(uploaded.read())
            st.success("OCR read complete. Review below, then click Apply to fill the form.")
            with st.expander("OCR text (for verification)"):
                st.text(ocr_text)
            st.markdown("**Parsed hours:**")
            st.json(ocr_hours)

            week_choice = st.radio("Apply to which week?", ["Week 1", "Week 2"], horizontal=True)
            if st.button("Apply OCR hours"):
                prefix = "w1" if week_choice == "Week 1" else "w2"
                for key, val in ocr_hours.items():
                    st.session_state[f"{prefix}_{key}"] = float(val or 0.0)
                st.success(f"Applied to {week_choice}. Scroll down to review/edit.")
        except Exception as exc:  # Keep it simple; show user-friendly error
            st.error(f"OCR failed: {exc}")

    st.info(
        "OCR runs in the app environment. Install requirements: `pip install pytesseract pillow` and the "
        "Tesseract engine for your OS or deployment platform."
    )

    with st.expander("Current payroll config (from config.json, overrides applied below)", expanded=False):
        st.json(apply_overrides(cfg))

    st.divider()

    w1 = week_inputs("Week 1", "w1")
    st.divider()
    w2 = week_inputs("Week 2", "w2")

    # Totals (auto-add)
    totals = {k: (w1[k] + w2[k]) for k in w1.keys()}

    st.divider()
    st.subheader("Totals (Week 1 + Week 2)")

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.metric("Total Regular", f"{totals['regular']:.2f} hrs")
        st.metric("Total OT", f"{totals['overtime']:.2f} hrs")
    with tc2:
        st.metric("Total PTO", f"{totals['pto']:.2f} hrs")
        st.metric("Total Holiday", f"{totals['holiday']:.2f} hrs")
    with tc3:
        st.metric("Total Evening Diff", f"{totals['evening_diff']:.2f} hrs")
        st.metric("Total Weekend Diff", f"{totals['weekend_diff']:.2f} hrs")
        st.metric("Total Night Diff", f"{totals['night_diff']:.2f} hrs")

    effective_cfg = apply_overrides(cfg)
    out = compute(effective_cfg, totals)

    eligible_cap = float(st.session_state["pto_policy"].get("eligible_hours_cap", 80.0))
    eligible_hours = min(totals["regular"] + totals["overtime"] + totals["holiday"] + totals["pto"], eligible_cap)
    pto_detail = compute_pto_accrual(
        st.session_state["pto_policy"]["years_of_service"],
        eligible_hours,
        st.session_state["pto_policy"]["pay_periods"],
        st.session_state["pto_policy"]["annual_pto_override"],
    )

    st.divider()
    st.metric("Estimated Net Pay", f"${out['net']:,.2f}")
    st.write(f"Estimated Gross Pay: **${out['gross']:,.2f}**")
    st.metric(
        "Expected PTO accrued this paycheck",
        f"{pto_detail['accrued']:.2f} hrs",
        help=(
            f"Eligible hours counted: {pto_detail['eligible_hours_used']:.2f}. "
            f"Rate per hour: {pto_detail['rate_per_hour']:.4f}. "
            f"Per-pay cap: {pto_detail['per_pay_cap']:.2f} hrs. "
            f"Annual cap: {pto_detail['annual_cap']:.0f} hrs."
        ),
    )

    st.markdown("## Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Earnings")
        st.write(out["earnings"])
        st.markdown("### Taxable wages")
        st.write(out["wages"])
    with c2:
        st.markdown("### Pre-tax deductions")
        st.write(out["pretax"])
        st.markdown("### Taxes (estimated)")
        st.write(out["taxes"])
