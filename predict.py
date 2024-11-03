from datetime import datetime
import numpy as np

def predict_travel_time(model, scaler, your_stop, average_dwell_time, average_travel_time):
    now = datetime.now()
    current_hour = now.hour
    current_day_of_week = now.weekday()

    new_record = np.array([[your_stop, current_hour, current_day_of_week, average_dwell_time, average_travel_time]])
    new_record_scaled = scaler.transform(new_record)
    prediction = model.predict(new_record_scaled)

    minutes = prediction[0] // 60
    seconds = prediction[0] % 60
    print("╔════════════════════════════════╗")
    print(f"║ {int(minutes[0])} min {int(seconds[0])} sec")
    print("╚════════════════════════════════╝")
