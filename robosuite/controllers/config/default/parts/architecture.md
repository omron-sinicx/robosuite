``` mermaid
flowchart TB
    %% LEFT SIDE: Soft Wrist + OSC_POSITION (Baseline - Working)
    subgraph Soft["Soft Wrist + OSC_POSITION (Working ✓)"]
        direction TB
        A1["Policy Action<br/>6D: Δpos, Δori"]
        A2["OSC_POSITION Controller<br/>kp=150<br/>damping=10<br/>Single Layer"]
        A3["Output:<br/>Joint torques"]
        A4["MuJoCo Simulation"]
        A5["Observations:<br/>Proprioception + gripper pose<br/>~30D"]
        A6["Metrics:<br/>Success: 20-60%<br/>Ep Length: 180-200"]
        A7["Failure Mode:<br/>Timeout, rare issues"]
        A8["Passive Compliance:<br/>Soft wrist flex joints<br/>dampen oscillations"]
        
        A1 --> A2
        A2 --> A3
        A3 --> A4
        A4 --> A5
        A5 --> A1
        A2 -.-> A8
        A5 --> A6
        A6 --> A7
    end
    
    %% RIGHT SIDE: Rigid Wrist + FDCC (Current - Failing)
    subgraph Rigid["Rigid Wrist + FDCC (Failing ✗)"]
        direction TB
        B1["Policy Action<br/>12D: Δpose(6) + wrench(6)"]
        B2["FDCC Outer Controller<br/>stiffness=500<br/>kp=0.05, kd=0<br/>error_scale=0.005"]
        B3["OSC_POSE Inner Controller<br/>kp=2000 → 150 tuned<br/>Two-Layer Cascade!"]
        B4["Output:<br/>Joint torques"]
        B5["MuJoCo Simulation"]
        B6["Observations:<br/>Proprioception + gripper pose<br/>~30D"]
        B7["Metrics:<br/>Success: 0%<br/>Ep Length: 30-50"]
        B8["Failure Mode:<br/>going_away_from_goal<br/>70-90% episodes"]
        B9["Software Compliance Only:<br/>No mechanical damping"]
        B10["The Catch-22:<br/>Weak gains → can't reach goal<br/>Strong gains → chaotic"]
        
        B1 --> B2
        B2 --> B3
        B3 --> B4
        B4 --> B5
        B5 --> B6
        B6 --> B1
        B2 -.-> B9
        B2 -.-> B10
        B6 --> B7
        B7 --> B8
    end
    
    %% Key Differences
    A1 -.->|6D vs 12D| B1
    A2 -.->|Single layer vs Cascaded| B3
    A8 -.->|Passive vs Software| B9
    A6 -.->|20-60% vs 0%| B7
    A7 -.->|Timeout vs Early term| B8
    
    %% Add note box
    Note["Key Problem:<br/>Cascaded gains = 500×0.005×2000 = 5000<br/>FDCC (compliant) + OSC (stiff) = Conflict<br/>Original FDCC = Single layer to hardware<br/>Robosuite FDCC = Two layers (not in paper)"]
    
    B3 -.-> Note
    
    %% Styling
    classDef working fill:#d4f7d4,stroke:#2ecc40,stroke-width:2px
    classDef failing fill:#ffd6d6,stroke:#e74c3c,stroke-width:2px
    classDef warning fill:#fff9d6,stroke:#f1c40f,stroke-width:2px
    
    class A1,A2,A3,A4,A5,A6,A7,A8 working
    class B1,B4,B7,B8 failing
    class B2,B3,B5,B6,B9,B10 warning
    class Note warning
```