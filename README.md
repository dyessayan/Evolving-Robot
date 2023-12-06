# Evolving-Robot
Developing a Physics Simulator 
Robot Evolution Simulation Project
Overview
This project focuses on the development and evolution of robotic locomotion through a simulated environment. It is structured into three distinct phases, each building on the successes and learnings of the previous one.

Phase A: Physics Simulator
Objective: Develop a foundational physics simulator to demonstrate basic robotic interactions with a simulated environment.
Demonstration: A cube "breathing" through sinusoidal dimension changes and bouncing on a flat plane, testing the physics engine's accuracy and responsiveness.
Phase B: Evolution of Fixed Morphology
Objective: Evolve a robot's locomotion with a fixed structure. The structure consists of connected cubes or tetrahedrons, with locomotion simulated by changing the rest length of the springs according to a sinusoidal function.
Approach: Employ a global frequency parameter to modulate spring lengths and simulate robot movement across the plane. The evolutionary process optimizes spring constants to maximize the distance traveled, indicative of locomotive efficiency.
Phase C: Evolution of Variable Morphology
Objective: Enhance the robot's morphology, allowing structural changes like adding and removing springs and masses, to evolve not just the locomotion but the robot's form.
Methodology: Adapt the code from Phase B to include morphological mutations, and use an evolutionary algorithm to simulate and select for robots that show the most significant movement over a number of cycles, thus evolving the fastest robot possible.
Learning and Adaptations
Progress Tracking: Learning curves and fitness dot plots are used to visualize and analyze the performance and efficiency of robots over generations.
Challenges: Refinement of the fitness function to better incentivize desired locomotion methods and prevent reliance on spring-loaded movements.
Performance: Continuous improvements in robot designs were achieved, as evidenced by the upward trend in fitness scores over successive generations.
Conclusions
This simulation project has laid the groundwork for understanding the complexities of robotic movement and the potential of evolutionary algorithms in navigating vast parameter spaces to find efficient locomotion strategies.
