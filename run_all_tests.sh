#!/bin/bash

# 1. Optimisation script
echo "######### 1. Optimisation script #########"
python tests_optimisation.py
echo "Optimisation script finished"
echo ""

# 2. Performance script
echo "######### 2. Performance script #########"
python tests_performance.py
echo "Performance script finished"
echo ""

# 3. Processing times - models
echo "######### 3. Processing times #########"
python tests_times.py
echo "Times script finished"
echo ""

# 4. XAI times 
echo "######### 4. XAI times #########"
python tests_times_xai.py
echo "XAI times script finished"
echo ""

# 5. Feature ranks
echo "######### 5. Feature ranks #########"
python tests_xai_ranks.py
echo "Feature ranks tests finished"
echo ""