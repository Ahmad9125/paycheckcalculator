#!/usr/bin/env python3
"""
Paycheck estimator (based on your most recent stub + config.json)

Model:
- Gross earnings from hours + differentials
- 401(k) % reduces federal/state taxable wages but NOT FICA
- Cafeteria pre-tax benefits (health/dental/vision) reduce income-tax wages AND FICA wages
- FICA: OASDI (6.2%) + Medicare (1.45%) on FICA-taxable wages
- Federal + IL withholding estimated via effective rates in config.json

Edit config.json to fine-tune rates/deductions for future checks.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass
class Hours:
    regular: float = 0.0      # regular working hours
    overtime: float = 0.0     # overtime hours
    holiday: float = 0.0      # holiday premium hours (paid at holiday_hourly_rate)
    pto: float = 0.0          # PTO hours paid at base rate
    evening_diff: float = 0.0 # hours that receive evening differential
    weekend_diff: float = 0.0 # hours that receive weekend differential


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def money(x: float) -> str:
    return f"${x:,.2f}"


def compute_paycheck(h: Hours, cfg: dict) -> dict:
    base = float(cfg["base_hourly_rate"])
    ot_rate = float(cfg["overtime_hourly_rate"])
    hol_rate = float(cfg["holiday_hourly_rate"])
    diff = float(cfg["shift_diff_per_hour"])

    # Earnings
    earnings_regular = h.regular * base
    earnings_pto = h.pto * base
    earnings_holiday = h.holiday * hol_rate
    earnings_ot = h.overtime * ot_rate
    earnings_diff = (h.evening_diff + h.weekend_diff) * diff

    gross = earnings_regular + earnings_pto + earnings_holiday + earnings_ot + earnings_diff

    # Pre-tax deductions
    k401_pct = float(cfg["k401_percent"])
    k401_amt = gross * k401_pct

    pretax_benefits = cfg.get("pretax_benefits_per_paycheck", {})
    dental = float(pretax_benefits.get("dental_pretax", 0.0))
    health = float(pretax_benefits.get("health_pretax", 0.0))
    vision = float(pretax_benefits.get("vision_pretax", 0.0))
    cafeteria_total = dental + health + vision

    pretax_total = k401_amt + cafeteria_total

    # Taxable wages
    # 401k does NOT reduce FICA taxable wages; cafeteria benefits DO reduce FICA wages
    fica_wages = max(gross - cafeteria_total, 0.0)
    income_tax_wages = max(gross - pretax_total, 0.0)

    # Taxes
    fica_rates = cfg["fica_rates"]
    oasdi = fica_wages * float(fica_rates["oasdi"])
    medicare = fica_wages * float(fica_rates["medicare"])

    wh = cfg["withholding_estimates"]
    fed = income_tax_wages * float(wh["federal_effective_rate"])
    state = income_tax_wages * float(wh["state_il_effective_rate"])

    total_taxes = oasdi + medicare + fed + state
    net = gross - pretax_total - total_taxes

    return {
        "inputs": h,
        "earnings": {
            "regular": earnings_regular,
            "pto": earnings_pto,
            "holiday": earnings_holiday,
            "overtime": earnings_ot,
            "shift_diff": earnings_diff,
            "gross": gross,
        },
        "deductions": {
            "401k": k401_amt,
            "dental": dental,
            "health": health,
            "vision": vision,
            "pretax_total": pretax_total,
        },
        "taxable_wages": {
            "fica_wages": fica_wages,
            "income_tax_wages": income_tax_wages,
        },
        "taxes": {
            "oasdi": oasdi,
            "medicare": medicare,
            "federal_withholding_est": fed,
            "state_il_withholding_est": state,
            "taxes_total": total_taxes,
        },
        "net_pay_est": net,
    }


def prompt_float(label: str, default: float = 0.0) -> float:
    while True:
        s = input(f"{label} [{default}]: ").strip()
        if not s:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("Please enter a number (example: 12 or 12.5).")


def main():
    cfg = load_config()
    print("\nPaycheck estimator (based on config.json)\n" + "-" * 44)
    print(f"Base rate: {money(cfg['base_hourly_rate'])}/hr")
    print(f"OT rate:   {money(cfg['overtime_hourly_rate'])}/hr")
    print(f"Holiday:   {money(cfg['holiday_hourly_rate'])}/hr")
    print(f"Diff:      {money(cfg['shift_diff_per_hour'])}/hr")
    print(f"401(k):    {cfg['k401_percent']*100:.1f}%\n")

    h = Hours(
        regular=prompt_float("Regular hours"),
        overtime=prompt_float("Overtime hours"),
        holiday=prompt_float("Holiday premium hours"),
        pto=prompt_float("PTO hours"),
        evening_diff=prompt_float("Evening differential hours"),
        weekend_diff=prompt_float("Weekend differential hours"),
    )

    out = compute_paycheck(h, cfg)

    print("\nRESULTS\n" + "=" * 44)
    e = out["earnings"]
    d = out["deductions"]
    t = out["taxes"]
    w = out["taxable_wages"]

    print("\nEarnings")
    print(f"  Regular:    {money(e['regular'])}")
    print(f"  PTO:        {money(e['pto'])}")
    print(f"  Holiday:    {money(e['holiday'])}")
    print(f"  Overtime:   {money(e['overtime'])}")
    print(f"  Shift diff: {money(e['shift_diff'])}")
    print(f"  GROSS:      {money(e['gross'])}")

    print("\nPre-tax deductions")
    print(f"  401(k) ({cfg['k401_percent']*100:.1f}%): {money(d['401k'])}")
    print(f"  Dental:               {money(d['dental'])}")
    print(f"  Health:               {money(d['health'])}")
    print(f"  Vision:               {money(d['vision'])}")
    print(f"  PRE-TAX TOTAL:        {money(d['pretax_total'])}")

    print("\nTaxable wages")
    print(f"  FICA wages:           {money(w['fica_wages'])}")
    print(f"  Fed/State wages:      {money(w['income_tax_wages'])}")

    print("\nEmployee taxes (estimated)")
    print(f"  OASDI (6.2%):         {money(t['oasdi'])}")
    print(f"  Medicare (1.45%):     {money(t['medicare'])}")
    print(f"  Federal est:          {money(t['federal_withholding_est'])}")
    print(f"  IL State est:         {money(t['state_il_withholding_est'])}")
    print(f"  TAX TOTAL:            {money(t['taxes_total'])}")

    print("\nNET PAY (est.)")
    print(f"  {money(out['net_pay_est'])}\n")

    print("If your real withholding is consistently higher/lower,")
    print("edit config.json -> withholding_estimates.\n")


if __name__ == "__main__":
    main()
