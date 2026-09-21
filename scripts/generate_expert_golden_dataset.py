# generate_expert_golden_dataset.py
import json

EXPERT_DATASET = [
    {
        "category": "complex", "lang": "en",
        "question": "Patient is a 45-year-old HIV+ male, treatment-naive. Resistance testing shows K103N mutation (NNRTI resistance), no INSTI or PI resistance. eGFR is 45 mL/min. CD4=250, viral load 32,000 copies/mL. No HBV co-infection. What ART regimen do you recommend?",
        "reference": "Given the K103N NNRTI resistance mutation, all NNRTI-based regimens (Efavirenz, Nevirapine, Rilpivirine) are contraindicated. The preferred regimen is an INSTI-based combination: BIC/FTC/TAF (Bictegravir 50mg/Emtricitabine 200mg/Tenofovir alafenamide 25mg) as a single tablet once daily per NIH/HHS guidelines, or Dolutegravir 50mg QD plus TAF/FTC per GESIDA. TAF is mandatory over TDF because eGFR 45 mL/min is below the TDF threshold of 60 mL/min. TAF remains safe down to eGFR 15 mL/min. HLA-B*5701 testing is required before considering Abacavir. Monitor renal function every 3 months. Update genotype resistance testing before any future ART switch.",
        "source_file": "guidelines-adult-adolescent-arv.pdf", "source_page": 202
    },
    {
        "category": "complex", "lang": "en",
        "question": "Carlos is a 38-year-old HIV+ male, treatment-naive, CD4=180, VL=67,000. He was just diagnosed with pulmonary tuberculosis and will start Rifampin-based therapy (RHEZ). eGFR is 72 mL/min, no HBV co-infection. What ART should be initiated and what dose adjustments are needed?",
        "reference": "Initiate Dolutegravir (DTG) 50mg TWICE daily (not once daily) due to Rifampin's potent induction of UGT1A1 and CYP3A4, which reduces DTG plasma concentrations by approximately 54%. The NRTI backbone should be TAF 25mg/FTC 200mg once daily — TAF is preferred over TDF for its superior renal and bone safety profile. ART should be started within 2-8 weeks of TB treatment initiation; however, for CD4 <50, start within 2 weeks. Co-trimoxazole prophylaxis (TMP-SMX one DS tablet daily) is recommended regardless of CD4 count in HIV/TB co-infected patients. Monitor for Immune Reconstitution Inflammatory Syndrome (IRIS), which may cause paradoxical worsening of TB symptoms after ART initiation. An alternative to Rifampin is Rifabutin 300mg daily, which allows standard DTG 50mg once daily dosing due to weaker enzyme induction.",
        "source_file": "guidelines-adult-adolescent-arv.pdf", "source_page": 173
    },
    {
        "category": "complex", "lang": "en",
        "question": "Emma is a 6-year-old HIV+ girl, 18kg, treatment-naive since birth. CD4=420, VL=18,000 copies/mL. No opportunistic infections, no known resistance. No HBV co-infection. eGFR normal. What first-line ART regimen do you recommend?",
        "reference": "For a child weighing 18kg (between 14kg and <20kg), the preferred regimen per NIH/HHS pediatric guidelines is Dolutegravir 20mg once daily (using 5mg dispersible tablets) plus Abacavir 300mg/Lamivudine 150mg. Note: DTG dose is 20mg (not 25mg) for 14kg to <20kg weight band. HLA-B*5701 testing is mandatory before initiating Abacavir due to risk of severe hypersensitivity reaction. If HLA-B*5701 positive, use DTG plus TAF/FTC or RAL plus ABC/3TC as alternatives. Resistance testing (genotype) should be performed before ART initiation. Monitor VL at 4 weeks, 8-12 weeks, then every 3-6 months. Target: VL undetectable <50 copies/mL within 6 months. Formulation note: use pediatric dispersible tablets, not adult tablets.",
        "source_file": "guidelines-pediatric-arv.pdf", "source_page": 477
    },
    {
        "category": "complex", "lang": "en",
        "question": "Ahmed is a 52-year-old newly diagnosed HIV+ male, treatment-naive. CD4=78, VL=145,000 copies/mL. No opportunistic infections yet. eGFR 88 mL/min, no HBV. Lives in Madrid, Spain. What prophylaxis should be initiated and when should ART start?",
        "reference": "Initiate ART immediately regardless of CD4 count per NIH/HHS and GESIDA guidelines. Preferred regimen: DTG 50mg QD + TAF 25mg/FTC 200mg QD. For prophylaxis with CD4=78: (1) PCP prophylaxis: TMP-SMX 1 DS tablet daily — mandatory for CD4 <200, also covers Toxoplasma gondii if IgG positive; (2) Cryptococcal antigen (CrAg) serum screening mandatory before ART initiation for CD4 <100 — if positive, treat cryptococcal meningitis before starting ART to prevent fatal IRIS; (3) MAC prophylaxis (Azithromycin 1200mg weekly) is NOT yet required as CD4=78 is above the <50 threshold. Monitor closely for IRIS given very low CD4 and high VL. Perform genotype resistance testing and HLA-B*5701 before ART start. Screen for Toxoplasma IgG, CMV, and baseline cryptococcal antigen.",
        "source_file": "guidelines-adult-adolescent-oi.pdf", "source_page": 45
    },
    {
        "category": "complex", "lang": "en",
        "question": "Marie is a 41-year-old HIV+ woman, on EFV/TDF/FTC for 3 years, VL undetectable <50 for 2 years. She reports persistent CNS side effects: vivid dreams, difficulty concentrating, mood changes attributed to Efavirenz. CD4=680, eGFR 72 mL/min. No HBV, no resistance history. She wants to switch. What are the options?",
        "reference": "Multiple switch options are available for CNS side effects from Efavirenz. CRITICAL WARNING: Rilpivirine (RPV) is also an NNRTI and shares cross-resistance with Efavirenz — K103N mutation (common with EFV failure) causes RPV resistance. Genotype resistance testing is MANDATORY before switching to any NNRTI-based regimen. Preferred options (INSTI-based, no cross-resistance risk): (1) BIC/FTC/TAF (Biktarvy) — single tablet once daily, preferred by NIH/HHS 2024; (2) DTG 50mg QD + TAF/FTC — high barrier to resistance, excellent CNS tolerability; (3) DTG+3TC dual therapy if VL stably undetectable >6 months and no HBV. TDF should be replaced by TAF given eGFR 72. RPV/FTC/TAF is only acceptable if genotype confirms no NNRTI resistance AND historical VL was never >100,000 copies/mL. CNS side effects from EFV typically resolve within 4-8 weeks after switching to INSTI-based regimen.",
        "source_file": "guidelines-adult-adolescent-arv.pdf", "source_page": 621
    },
    {
        "category": "complex", "lang": "es",
        "question": "Isabel tiene 35 años, VIH+ naive, coinfectada con VHB (HBsAg positivo). CD4=290, carga viral 28,000 copias/mL. eGFR 85 mL/min, sin resistencias conocidas. ¿Qué régimen antirretroviral se recomienda y por qué es importante el manejo del VHB?",
        "reference": "El régimen recomendado es Dolutegravir (DTG) 50mg una vez al día más Tenofovir alafenamida (TAF) 25mg/Emtricitabina (FTC) 200mg una vez al día. TAF/FTC proporciona cobertura dual para VIH y VHB, siendo esencial en la coinfección. TAF es preferido sobre TDF por su mejor perfil de seguridad renal y ósea. NUNCA suspender los fármacos activos frente al VHB (TAF/FTC) sin cobertura alternativa, ya que puede producirse una reactivación grave del VHB con riesgo de insuficiencia hepática aguda. Monitorizar ADN-VHB y transaminasas (ALT/AST) cada 4 semanas. Si se requiere cambio de TAR en el futuro, mantener siempre al menos un fármaco activo frente al VHB. Realizar test de resistencias y HLA-B*5701 antes de iniciar. Objetivo: carga viral VIH <50 copias/mL y supresión de ADN-VHB.",
        "source_file": "GuiaGeSIDAPlanNacionalSobreElSidaRespectoAlTratamientoAntirretroviralEnAdultosInfectadosPorElVirusDeLaInmunodeficienciaHumanaActualizacionEnero2022.pdf", "source_page": 94
    },
    {
        "category": "simple", "lang": "en",
        "question": "Tom is a 28-year-old HIV-negative MSM with multiple sexual partners. eGFR 95 mL/min, HBV vaccinated, STI screening negative. He requests PrEP. What regimen and monitoring schedule do you recommend?",
        "reference": "Two PrEP options are available: (1) Daily PrEP: TDF 300mg/FTC 200mg (Truvada) once daily — standard regimen, effective >99% when adherent. Alternative: TAF 25mg/FTC 200mg (Descovy) for patients concerned about bone/renal effects, approved for MSM and transgender women. (2) On-demand PrEP (2-1-1 schedule, Ipergay protocol): 2 tablets 2-24h before intercourse, 1 tablet 24h after, 1 tablet 48h after — approved by GESIDA for MSM with infrequent exposure. Before initiating PrEP: confirm HIV-negative status within 7 days, assess renal function, screen for STIs and HBV/HCV. Monitoring schedule: HIV test every 3 months (mandatory), renal function every 6 months (TDF) or annually (TAF), STI screening every 3-6 months. Discontinue if eGFR <60 mL/min with TDF.",
        "source_file": "cdc-hiv-npep-guidelines.pdf", "source_page": 35
    },
    {
        "category": "complex", "lang": "en",
        "question": "Fatima is 31 years old, HIV+, 32 weeks pregnant. She has been on DTG+TDF/FTC for 18 months with previously undetectable VL. Her latest VL at 32 weeks is 1,250 copies/mL. eGFR has worsened to 58 mL/min. No HBV. Should she have a C-section? Should ART be adjusted? What about neonatal prophylaxis?",
        "reference": "Three issues require immediate attention: (1) ART adjustment: TDF is contraindicated with eGFR 58 mL/min (threshold <60). Switch immediately to DTG 50mg QD + TAF 25mg/FTC 200mg. TAF is renal-safe down to eGFR 15 mL/min. (2) Virologic failure investigation: VL 1,250 copies/mL after 18 months suggests adherence issues or resistance. Perform urgent genotype resistance testing. (3) Delivery planning: target undetectable VL before week 36. If VL remains >1,000 copies/mL at 36 weeks: plan elective C-section at 38 weeks with IV Zidovudine 2mg/kg loading dose then 1mg/kg/h during surgery. Neonatal prophylaxis: since maternal VL >50 copies/mL, newborn requires TRIPLE therapy (AZT + 3TC + NVP or RAL) for 6 weeks. Breastfeeding absolutely contraindicated. Monitor DTG timing with prenatal iron supplements (2h separation).",
        "source_file": "guidelines-perinatal.pdf", "source_page": 423
    },
    {
        "category": "complex", "lang": "en",
        "question": "Robert is a 58-year-old HIV+ male, stably undetectable on DTG+TAF/FTC for 3 years. He has type 2 diabetes on Metformin 1000mg twice daily, osteoporosis taking Calcium 500mg + Vitamin D supplements twice daily, and hypertension on Amlodipine 5mg. eGFR 62 mL/min. What drug interactions should be monitored and are any dose adjustments needed?",
        "reference": "Three critical drug interactions require management: (1) DTG + Calcium supplements: Dolutegravir chelates polyvalent cations including calcium. Take DTG at least 2 hours BEFORE or 6 hours AFTER calcium supplements. Vitamin D alone has no interaction with DTG — only calcium-containing products require separation. (2) DTG + Metformin: DTG inhibits renal OCT2, increasing Metformin plasma concentrations by approximately 79%, raising the risk of lactic acidosis. Consider reducing Metformin dose or switch to SGLT2 inhibitor or DPP-4 inhibitor. Monitor renal function every 3 months — Metformin contraindicated if eGFR <30. (3) TAF + Amlodipine: minor interaction, clinically insignificant. TAF appropriate at eGFR 62. No dose adjustment needed for DTG or TAF.",
        "source_file": "guidelines-adult-adolescent-arv.pdf", "source_page": 578
    },
    {
        "category": "complex", "lang": "en",
        "question": "David is a 44-year-old HIV+ male on RAL+TDF/FTC for 4 years with virologic failure. Genotype shows Q148H + G140S mutations (high-level INSTI resistance). No PI resistance detected. CD4=145, VL=8,500, eGFR 78 mL/min. No HBV. What salvage ART regimen do you recommend?",
        "reference": "Q148H + G140S mutations confer high-level resistance to ALL integrase inhibitors including Raltegravir, Dolutegravir, and Bictegravir — DTG twice daily is NOT adequate for this mutation pattern. Salvage regimen: Darunavir/ritonavir (DRV/r) 800mg/100mg once daily plus TAF 25mg/FTC 200mg, plus at least one additional active agent. Options: (1) Etravirine 200mg BID if no NNRTI resistance; (2) Maraviroc if CCR5-tropic (perform tropism testing); (3) For multi-class resistance: Fostemsavir (attachment inhibitor) or Ibalizumab (CD4 directed antibody). TAF preferred over TDF at eGFR 78. Goal: at least 2 fully active drugs in the salvage regimen. Consult HIV resistance specialist.",
        "source_file": "guidelines-adult-adolescent-arv.pdf", "source_page": 179
    },
]

with open("golden_dataset_expert.json", "w", encoding="utf-8") as f:
    json.dump(EXPERT_DATASET, f, indent=2, ensure_ascii=False)

print(f"Expert golden dataset: {len(EXPERT_DATASET)} pairs")
by_cat = {}
for q in EXPERT_DATASET:
    k = f"{q['category']}_{q['lang']}"
    by_cat[k] = by_cat.get(k, 0) + 1
for k, v in sorted(by_cat.items()):
    print(f"  {k}: {v}")