Reinforcement Learning Agents Fighting

A Pygame environment where two reinforcement learning agents compete against each other. This repo uses pretrained policy files so you can run the game and evaluate agents without training.

Features
- Pygame based 2D fighting environment
- Runs with pretrained PyTorch policy files
- Frame by frame observation and prediction overlay

Getting Started

Prerequisites
- Python 3.9 or higher
- Install dependencies
pip install -r requirements.txt

Run the Game with Pretrained Policies
1. Clone the repo
git clone https://github.com/your-username/RL-Agents-Fighting.git
cd RL-Agents-Fighting

2. Place pretrained policy files
Download from Drive and put them in the location of the file main.py

3. Run the main game that loads the pretrained policies

If your main script uses different argument names replace them accordingly

Pretrained Models and Large Assets
- Pretrained policies, large sprites, and other big files are stored in assets
- Some files may be large. If assets are not in the repo download them from the provided Google Drive link and place them in the assets folder

Project Structure
assets/                (sprites, environment maps)
Game_Module.py         (Game script)
main.py                (runs the game and loads pretrained policies)
README.md

Notes
- This project is for educational use to explore reinforcement learning and game AI
- The repo does not include training scripts. Only pretrained policy files are supported
- Models and results may vary depending on system and parameters
- To Get the Pretrained Policies: Download them from https://drive.google.com/file/d/1KD2HBOCUmtaWHyo7ZKXw57WFNtnBcShm/view?usp=drive_link

Contact
GitHub: https://github.com/L0stxRising
Questions and pull requests are welcome
