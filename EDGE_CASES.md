# Edge Cases

Running list of extraction failure patterns discovered during manual auditing (2026-02-22).
Based on "both miss" records from V5 vs V5a full adversarial runs.

---

## 1. Gold truth is wrong (model is actually correct)
- **TX Dallas 2018 Police**: Gold $493M, model $465M — model answer is correct
- **TX Mexia 2023 GF**: Gold $7.8M, model $8.4M — model answer is correct
- **CA Santa Cruz 2022 Police**: Gold answer is wrong, expected is correct
- **MD Rockville 2022 Police**: Gold is total, should be GF only — model's $13.2M may be right
- **Action**: Fix these in the validation table

## 2. Transfers included/excluded (ambiguous gold truth)
No consistent rule on whether "total GF expenditures" includes transfers.
- **TX Irving 2018 GF**: $211M (without) vs $232M (with transfers)
- **TX San Antonio 2021 GF**: $1.26B (without) vs $1.29B (with transfers)
- **WA Tukwila 2019 GF**: $65.9M (with) vs $59.5M (without transfers)
- **WA Tukwila 2022 GF**: $66.8M (with) vs $58.9M (without transfers)
- **SD Aberdeen 2019 GF**: Gold "incl transfers" — model gave different number
- **Action**: Decide on a consistent rule (with or without transfers) and update gold values

## 3. Wrong fiscal year (correct page, wrong column)
Model picks an adjacent year from the same table — very common with multi-year tables.
- **AK Juneau 2023 Police**: Pulled FY24 instead of FY23 (biennial budget)
- **AL Dothan 2020 GF**: Wrong year (biennial)
- **CA Santa Cruz 2018 GF**: Pulled 2017 instead of 2018
- **CA Santa Cruz 2019 Police**: Pulled 2018 instead of 2019
- **CO Golden 2023 GF**: Pulled proposed 2024 instead of 2023 (biennial)
- **OR Salem 2019 GF**: Pulled 2018 adopted
- **OR Salem 2021 GF**: Pulled 2020 instead of 2021
- **SD Aberdeen 2019 GF**: Pulled FY18
- **TN Germantown 2018 Police**: Pulled FY19 projected
- **Action**: Prompt engineering (emphasize fiscal year), or training data augmentation with multi-year tables

## 4. Wrong fund or category
### Revenue vs expenditure
- **AZ Yuma 2019 GF**: Pulled revenues (wrong page)
- **CA Modesto 2019 GF (A)**: Pulled revenues

### All-funds vs General Fund
- **CA Santa Rosa 2023 Police**: Pulled all-funds ($73M) instead of GF ($68M)
- **PA State College 2018 Police**: Total, not GF
- **PA State College 2021 Police**: Total, not GF

### Wrong department
- **AR Bentonville 2021 Police**: Pulled fire department (fire > police, so maybe largest?)
- **AR Bentonville 2022 Police**: Same — fire instead of police

### Sub-department or line item vs total
- **CA Santa Cruz 2018 Police**: Personnel Services, not total
- **NY Troy 2019 Police**: Employee benefits only
- **NY Troy 2022 Police**: Personnel Services, not total
- **WI Kenosha 2020 Police**: Police patrol only

### Missing education
- **MA Cambridge 2022 GF**: $447M excludes education — gold $735M includes it

## 5. Proposed/recommended vs adopted
- **NY Long Beach 2019 GF**: Pulled proposed instead of adopted
- **NY Newburgh 2022 Police (B)**: "Requested stage" budget

## 6. Multi-page tables / parsing failure
- **TN Germantown** (2018, 2021, 2022, 2023 Police): Can't read these tables at all. Row labels on previous page, amounts on target page. Both runs way off every year.
- **WI Kenosha** (2018, 2020 Police): Way off, value not on same page
- **Action**: Aryn parser may handle multi-page tables; include previous page as context

## 7. Rounding
- **PA Norristown 2019 GF**: Gold $34,545,082.99, model $34,545,083. Off by $1.
- **Action**: Tolerance-based matching or consistent rounding in gold values

## 8. Close / acceptable
- **AK Anchorage 2022 GF**: $539.7M vs $539.9M — unclear difference
- **CA Los Angeles 2018 GF**: Departmental vs budgetary departments — acceptable
- **CA Modesto 2019 GF (B)**: Operating not total — acceptable
- **CA Santa Cruz 2018 Police (B)**: Police "subtotal general fund" — acceptable
- **MN Fridley 2020 GF**: $17.69M vs $17.96M — possibly correct?

## 9. Way off / unknown
- **CO Aurora** (2019, 2020, 2022 Police): Pulling internal service funds, risk management, HR — completely wrong section of budget
- **DE Newark** (2019, 2022, 2023 GF): Pulling utility purchases — wrong section
- **CA Vallejo 2023** (GF and Police): Both close but unexplained differences
- **CA San Francisco 2020 GF**: "Total Departmental Sources" vs "Total General City Responsibilities" — needs investigation
- **NY Newburgh 2022 Police (A)**: No idea where $22.9M comes from
- **UT Ogden 2018 Police**: Way off, right page — parsing issue?
- **OR Hermiston 2020 Police**: Both answers close but wrong

---

## Summary of actionable fixes
1. **Fix gold truth** for Dallas, Mexia, Santa Cruz 2022, Rockville (4 records)
2. **Decide transfers rule** — with or without? Update gold consistently (5+ records)
3. **Wrong year is the biggest category** — prompt/training improvement needed (9 records)
4. **Parsing failures** (Germantown, Kenosha) — need better parser or adjacent-page context
5. **Wrong fund/category** — prompt already says "General Fund" and "adopted" but model ignores it
