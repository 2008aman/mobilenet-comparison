# mobilenet-comparison
Benchmarking MobileNetV1–V3 for IoT object recognition

setup : use requirements.txt to setup your local environment. 

Results from my local system


🔹 Evaluating MobileNetV1 ...
Top Prediction: Samoyed (99.94%)
Inference Time: 0.517s
Parameters: 4.25M

🔹 Evaluating MobileNetV2 ...
Top Prediction: Samoyed (94.25%)
Inference Time: 0.796s
Parameters: 3.54M

🔹 Evaluating MobileNetV3Small ...
Top Prediction: Samoyed (83.48%)
Inference Time: 0.757s
Parameters: 2.55M

🔹 Evaluating MobileNetV3Large ...
Top Prediction: Samoyed (91.57%)
Inference Time: 1.029s
Parameters: 5.51M

📊 COMPARISON SUMMARY:
Model               Confidence (%)    Time (s)  Params (M)
MobileNetV1         99.94             0.517     4.25
MobileNetV2         94.25             0.796     3.54
MobileNetV3Small    83.48             0.757     2.55
MobileNetV3Large    91.57             1.029     5.51
