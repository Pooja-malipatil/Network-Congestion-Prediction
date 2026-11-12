# 4G Network Congestion Prediction (Machine Learning)

Predicts which type of congestion a 4G cell has: `NC`, `4G_RAN_CONGESTION`,
`4G_BACKHAUL_CONGESTION` or `3G_BACKHAUL_CONGESTION`.

## Files
- `train_model.py` - trains Random Forest and Gradient Boosting, picks the best, saves results
- `notebooks/4G_Network_Congestion_Analysis.ipynb` - same steps with graphs and explanations
- `app.py` - Streamlit dashboard (explore data, important features, try the model)
- `data/` - `train_upd.csv` (with answers), `test_upd.csv` (without answers)

## How to run
```
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
`train_model.py` creates the `outputs/` files (predictions, saved model, feature importance)
that the dashboard uses, so run it before the dashboard.

## Results
| Model | Accuracy (5-fold CV) |
|---|---|
| Random Forest | about 74% |
| Gradient Boosting (best) | about 79% |

The model is best at spotting `NC` and weakest at separating 4G backhaul from 4G RAN congestion.