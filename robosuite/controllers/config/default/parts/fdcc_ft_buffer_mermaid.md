# How `ft_buffer_size` is Used During Training (FDCC Controller)

```mermaid
flowchart TD
    subgraph Training Loop
        A[Environment Step] --> B[Controller Step (FDCC)]
        B --> C[Read Force/Torque Sensor]
        C --> D[Append to FT Buffer (size=25)]
        D --> E[Compute Filtered FT Value (e.g., mean or median)]
        E --> F[Use Filtered FT in Control Law]
        F --> G[Compute Action (torque/position)]
        G --> H[Apply Action to Robot]
    end
    
    subgraph FDCC Controller
        D
        E
        F
    end
    
    note1((Note: OSC_POSE inner controller does not use ft_buffer_size))
```

**Explanation:**
- At each environment step during training, the FDCC controller reads the force/torque (FT) sensor.
- The new FT reading is appended to a buffer of the last 25 readings (`ft_buffer_size=25`).
- The controller computes a filtered value (e.g., mean or median) from the buffer to reduce noise.
- This filtered FT value is used in the FDCC control law to compute the next action.
- The action is then applied to the robot/environment.
- The inner OSC_POSE controller does not use `ft_buffer_size`.
