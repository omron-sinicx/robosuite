# Polygon Mesh Generator

This tool generates random polygon meshes.

## Generation

Generate random polygon meshes with the following command:

```bash
python3 main.py -o <output_dir>
```

### Options

- `-o, --output_dir`: Output directory for the generated meshes (default: `./output`)
- `-minv`: Minimum number of vertices for polygon meshes (default: 3)
- `-maxv`: Maximum number of vertices for polygon meshes (default: 10)
- `-c, --count`: Number of meshes to generate for each vertex count (default: 1)
- `--tolerance`: Distance tolerance between peg and hole in millimeters
- `--seed`: Random seed for reproducible generation

For example, to generate meshes with vertices from 4 to 6 and each count of 10, run:

```bash
python3 main.py -o ./output -minv 4 -maxv 6 -c 10
```

## Integration with the soft-peg-in-hole environment

Follow these steps to use the generated polygon meshes in the soft-peg-in-hole environment:

1. Generate the polygon meshes:
   ```bash
   cd robosuite/scripts/generate_polygon_mesh
   python3 main.py -o ./output
   ```

2. Create symbolic links to avoid redundancy:
   ```bash
   # Link the peg meshes to the grippers directory
   cd ../../models/assets/grippers/meshes
   ln -s ../../../../scripts/generate_polygon_mesh/output/meshes/peg <shape_dir>

   # Link the hole meshes to the objects directory
   cd ../../objects/meshes
   ln -s ../../../../scripts/generate_polygon_mesh/output/meshes/hole <shape_dir>
   ```
   Note: Replace `<shape_dir>` with your desired directory name. Use the same name for both links.

3. Update the mesh file paths in the XML configuration files:
   - Edit `robosuite/models/assets/grippers/robotiq_gripper_85_soft.xml`
   - Edit `robosuite/models/assets/objects/alan-xml.xml`
   - Replace the default path `meshes/alan-poly/poly4_0/part_0.stl` with a existed mesh, such as `meshes/<shape_dir>/poly4_0/part_0.stl`

This setup ensures that the environment can find the corresponding peg and hole meshes for simulation.

4. Run the soft-peg-in-hole environment:
    - Assign the environment parameters `peg_shape=custom` and `mesh_dir=<shape_dir>`
   ```bash
   cd robosuite/demos
   python3 test_keyboard_control.py --shape custom --mesh_dir <shape_dir>
   ```
