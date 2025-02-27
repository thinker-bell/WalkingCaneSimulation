import pybullet as p
import pybullet_data
import numpy as np
import time

# Connect to PyBullet in GUI mode
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()

# Parameters for the heightfield terrain.
# This terrain is WAY to textured to run the cane on. 
numRows = 256
numCols = 256
heightScale = 1.0
meshScale = [0.05, 0.05, heightScale]
heightfieldData = [np.random.uniform(-0.1, 0.1) for _ in range(numRows * numCols)]

'''
# DUNES TERRAIN, GENTLE WAVES IN THE FLOOW
numRows = 128
numCols = 128
heightScale = 0.8
meshScale = [0.2, 0.2, 1.2]
heightfieldData = np.sin(np.linspace(0, np.pi * 4, 128 * 128))
heightfieldTextureScaling = 3.0
'''





terrainShape = p.createCollisionShape(
    shapeType=p.GEOM_HEIGHTFIELD,
    meshScale=meshScale,
    heightfieldTextureScaling=(numRows - 1) / 2,
    heightfieldData=heightfieldData,
    numHeightfieldRows=numRows,
    numHeightfieldColumns=numCols
)

terrain = p.createMultiBody(0, terrainShape)

# Optionally change the visual appearance (if you have a texture file)
try:
    textureId = p.loadTexture("grass.png")
    p.changeVisualShape(terrain, -1, textureUniqueId=textureId)
except Exception as e:
    print("Texture load failed:", e)

# Simulation loop that keeps running until you press Enter in the console.
print("Simulation running. Press Enter to exit.")
while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
    # Check if user has pressed Enter.
    if input():
        break

p.disconnect()
