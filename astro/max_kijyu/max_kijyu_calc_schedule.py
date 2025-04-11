from datetime import datetime, timedelta
import pandas as pd

# 会議の基本情報
start_time = datetime.strptime("10:00", "%H:%M")
end_time = datetime.strptime("17:00", "%H:%M")
lunch_break_start = datetime.strptime("12:00", "%H:%M")
lunch_break_end = datetime.strptime("13:00", "%H:%M")
break_duration = timedelta(minutes=10)  # 休憩時間（10分）
presentation_duration = timedelta(minutes=10)  # 1人の発表時間（発表7分 + 質疑・交代3分 = 10分）

# 計算
total_time = (end_time - start_time) - (lunch_break_end - lunch_break_start)
max_presentations = total_time // presentation_duration

# 休憩の考慮（午前・午後それぞれ1回、10分ずつ）
break_times = 2 * break_duration
max_presentations_after_breaks = (total_time - break_times) // presentation_duration

# スケジュール作成
schedule = []
current_time = start_time

for i in range(max_presentations_after_breaks):
    if current_time == lunch_break_start:
        schedule.append([lunch_break_start.strftime("%H:%M"), lunch_break_end.strftime("%H:%M"), "昼休憩"])
        current_time = lunch_break_end

    if i == max_presentations_after_breaks // 2:
        break_end_time = current_time + break_duration
        schedule.append([current_time.strftime("%H:%M"), break_end_time.strftime("%H:%M"), "休憩"])
        current_time = break_end_time

    end_presentation_time = current_time + presentation_duration
    schedule.append([current_time.strftime("%H:%M"), end_presentation_time.strftime("%H:%M"), f"発表{i+1}"])
    current_time = end_presentation_time

# DataFrameに変換
df_schedule = pd.DataFrame(schedule, columns=["開始時間", "終了時間", "内容"])

# 結果を表示
print(df_schedule)
