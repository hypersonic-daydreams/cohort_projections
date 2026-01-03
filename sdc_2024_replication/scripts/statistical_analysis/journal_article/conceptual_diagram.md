# Conceptual Diagram: Policy Mechanisms and Migration Flows

## Overview

This document provides a conceptual diagram showing the causal pathway from federal immigration policy to observed North Dakota net international migration. The diagram operationalizes the "faucet-pipe-stickiness" framework from the ChatGPT policy analysis (ADR-021 Recommendation #8).

**Key Insight**: ND's foreign-born inflow is governed by:
```
Inflow = Federal "Faucet" x State "Pipe Diameter" x Legal "Stickiness"
```

---

## Primary Conceptual Diagram

```mermaid
flowchart TB
    subgraph Federal["Federal Policy Layer (Supply/Faucet)"]
        direction TB
        CEILING["Refugee Ceilings<br/>(Presidential Determination)"]
        PAROLE["Parole Programs<br/>(OAW, U4U, CHNV)"]
        TRAVEL["Travel Restrictions<br/>(EO, Proclamations)"]
        PROCESSING["Processing Capacity<br/>(Consular, USCIS)"]
    end

    subgraph State["ND Capacity Layer (Allocation/Pipe)"]
        direction TB
        RECEPTION["Reception Agencies<br/>(LSSND → Global Refuge)"]
        LABOR["Labor Demand<br/>(Oil, Healthcare, Ag)"]
        SECONDARY["Secondary Migration<br/>(FB from other states)"]
    end

    subgraph Status["Status Durability Layer (Retention/Stickiness)"]
        direction TB
        DURABLE["Durable Status<br/>(Refugee, LPR, SIV)"]
        TEMP["Temporary Status<br/>(Parole 2-year)"]
        REGULARIZE["Regularization<br/>(Adjustment legislation)"]
    end

    subgraph Observed["Observed Population"]
        direction TB
        YT["Y_t: PEP Net International Migration"]
        YDUR["Y_t^dur: Durable Component"]
        YTEMP["Y_t^temp: Temporary Component"]
    end

    %% Federal to State flows
    CEILING --> RECEPTION
    PAROLE --> RECEPTION
    TRAVEL -.->|"restricts"| RECEPTION
    PROCESSING -.->|"constrains"| RECEPTION

    %% State internal flows
    LABOR --> SECONDARY
    RECEPTION --> DURABLE
    RECEPTION --> TEMP

    %% Status transitions
    DURABLE --> YDUR
    TEMP -->|"if regularized"| REGULARIZE
    REGULARIZE --> YDUR
    TEMP -->|"if not regularized"| YTEMP
    TEMP -.->|"attrition (years 2-4)"| EXIT["Exit/Emigration"]

    %% Secondary migration
    SECONDARY --> YT

    %% Aggregation
    YDUR --> YT
    YTEMP --> YT

    %% Styling
    classDef federal fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef state fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef status fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef observed fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef exit fill:#ffebee,stroke:#c62828,stroke-width:2px

    class CEILING,PAROLE,TRAVEL,PROCESSING federal
    class RECEPTION,LABOR,SECONDARY state
    class DURABLE,TEMP,REGULARIZE status
    class YT,YDUR,YTEMP observed
    class EXIT exit
```

---

## Simplified Linear Diagram

For presentation contexts requiring a simpler view:

```mermaid
flowchart LR
    subgraph A["Federal Policy<br/>(Supply/Faucet)"]
        FP["Ceilings<br/>Parole Programs<br/>Travel Rules<br/>Processing"]
    end

    subgraph B["ND Capacity<br/>(Allocation/Pipe)"]
        NC["Reception Agencies<br/>Labor Demand<br/>Sponsor Networks"]
    end

    subgraph C["Status Durability<br/>(Retention/Stickiness)"]
        SD["Refugee: High retention<br/>Parole: Cliff at years 2-4"]
    end

    subgraph D["Observed"]
        OBS["Y_t: PEP Net<br/>International Migration"]
    end

    A -->|"x"| B -->|"x"| C -->|"="| D
```

---

## Regime-Specific Pathway Diagram

Shows how the pathway operates differently across policy regimes:

```mermaid
flowchart TB
    subgraph Expansion["Expansion Regime (2010-2016)"]
        E_FED["High Ceilings<br/>(70K-110K)"]
        E_ND["LSSND Active<br/>Strong Reception"]
        E_STATUS["92% Refugee<br/>(Durable)"]
        E_OUT["Mean: 1,289/yr"]
        E_FED --> E_ND --> E_STATUS --> E_OUT
    end

    subgraph Restriction["Restriction Regime (2017-2020)"]
        R_FED["Low Ceilings<br/>(18K-45K)<br/>+ Travel Bans"]
        R_ND["LSSND Active<br/>Reduced Flow"]
        R_STATUS["~100% Refugee<br/>(Durable)"]
        R_OUT["Mean: 1,197/yr"]
        R_FED --> R_ND --> R_STATUS --> R_OUT
    end

    subgraph Volatility["Volatility Regime (2021-2024)"]
        V_FED["High Ceilings<br/>(62.5K-125K)<br/>+ Parole Surge"]
        V_ND["LSSND Closed<br/>Global Refuge<br/>Rebuilding"]
        V_STATUS["7% Refugee<br/>93% Parole/Temp"]
        V_OUT["Mean: 3,284/yr<br/>(but fragile)"]
        V_FED --> V_ND --> V_STATUS --> V_OUT
    end

    %% Styling
    classDef expansion fill:#c8e6c9,stroke:#388e3c
    classDef restriction fill:#ffcdd2,stroke:#c62828
    classDef volatility fill:#fff9c4,stroke:#f9a825

    class E_FED,E_ND,E_STATUS,E_OUT expansion
    class R_FED,R_ND,R_STATUS,R_OUT restriction
    class V_FED,V_ND,V_STATUS,V_OUT volatility
```

---

## Status Transition Hazard Diagram

Illustrates the "parole cliff" concept:

```mermaid
flowchart LR
    subgraph Entry["Arrival Year 0"]
        REF_IN["Refugee Arrival"]
        PAR_IN["Parole Arrival"]
    end

    subgraph Y1["Year 1"]
        REF_1["Refugee<br/>Status Stable"]
        PAR_1["Parole<br/>Status Active"]
    end

    subgraph Y2["Year 2 (Cliff)"]
        REF_2["Refugee → LPR<br/>(Adjustment)"]
        PAR_2A["Parole → "]
        PAR_2B["Extension?"]
        PAR_2C["Asylum?"]
        PAR_2D["Exit?"]
    end

    subgraph Y5["Year 5+"]
        REF_5["LPR → Citizen<br/>(Naturalization)"]
        PAR_5A["Adjusted → LPR"]
        PAR_5B["Out of Status"]
    end

    REF_IN --> REF_1 --> REF_2 --> REF_5
    PAR_IN --> PAR_1 --> PAR_2A
    PAR_2A --> PAR_2B
    PAR_2A --> PAR_2C
    PAR_2A --> PAR_2D
    PAR_2B -.-> PAR_5B
    PAR_2C --> PAR_5A
    PAR_2D -.-> EXIT["Emigration/<br/>Attrition"]

    %% Styling
    style REF_IN fill:#c8e6c9
    style REF_1 fill:#c8e6c9
    style REF_2 fill:#a5d6a7
    style REF_5 fill:#81c784
    style PAR_IN fill:#fff9c4
    style PAR_1 fill:#fff9c4
    style PAR_2A fill:#ffcc80
    style PAR_2B fill:#ffcc80
    style PAR_2C fill:#ffcc80
    style PAR_2D fill:#ffcc80
    style PAR_5A fill:#a5d6a7
    style PAR_5B fill:#ffcdd2
    style EXIT fill:#ef9a9a
```

---

## Module Mapping Diagram

Shows how the conceptual framework maps to analysis modules:

```mermaid
flowchart TB
    subgraph Conceptual["Conceptual Layer"]
        C1["Federal Supply"]
        C2["ND Capacity"]
        C3["Status Durability"]
    end

    subgraph Modules["Analysis Modules"]
        M2["Module 2<br/>Time Series + Breaks"]
        M4["Rec #4<br/>Policy Regime R_t"]
        M7["Module 7<br/>Causal Inference"]
        M7B["Rec #3<br/>LSSND Synth Control"]
        M8["Module 8<br/>Duration Analysis"]
        M8B["Rec #2<br/>Status-Aware Hazards"]
        M9["Module 9<br/>Scenario Modeling"]
        M9B["Rec #6<br/>Policy-Lever Scenarios"]
    end

    subgraph Outputs["Outputs"]
        O1["Structural Break Dates"]
        O2["Capacity Effect Estimate"]
        O3["Status-Specific Forecasts"]
        O4["Policy Scenario Ranges"]
    end

    C1 --> M2 & M4 --> O1
    C2 --> M7 & M7B --> O2
    C3 --> M8 & M8B --> O3
    M4 & M7B & M8B --> M9 & M9B --> O4
```

---

## Usage Notes

### Rendering Mermaid Diagrams

1. **GitHub/GitLab**: Native rendering in markdown files
2. **VS Code**: Mermaid Preview extension
3. **Jupyter**: mermaid-python package
4. **Static Export**: Use mermaid-cli (`mmdc`) to export to PNG/SVG/PDF

### Export Command Example
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Export to PNG
mmdc -i conceptual_diagram.md -o conceptual_diagram.png -t dark

# Export to SVG
mmdc -i conceptual_diagram.md -o conceptual_diagram.svg
```

### LaTeX Integration

For LaTeX documents, use the TikZ version in `conceptual_diagram.tex` or export Mermaid to PDF/PNG and include as figure.

---

## Diagram Interpretation Guide

### Color Coding

| Color | Meaning |
|-------|---------|
| Blue | Federal policy layer |
| Purple | State capacity layer |
| Green | Durable status / positive outcomes |
| Yellow/Orange | Temporary status / uncertain outcomes |
| Red | Attrition / exit |

### Arrow Types

| Arrow | Meaning |
|-------|---------|
| Solid (-->) | Primary causal pathway |
| Dashed (-..->) | Constraint or negative effect |
| Multiplicative (x) | Interaction between factors |

---

*Document Version: 1.0*
*Created: 2026-01-02*
*ADR Reference: ADR-021 Recommendation #8*
