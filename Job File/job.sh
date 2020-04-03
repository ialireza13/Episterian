#!/bin/bash -l
#PBS -l nodes=1:ppn=1
#PBS -m abe
#pbs -M sinasajjadi@protonmail.com
#PBS -q LSPR
cd /home/seyedebrahim.saj.physics.sharif/agentBasedSpreading/Src
/share/apps/Anaconda/anaconda3.6/bin/python3.6 runScenario.py

