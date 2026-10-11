# MAS5 V5.8 — Closed-loop adaptive Role probing (one-MAS pilot)

זו **הפעלה אדפטיבית אמיתית מבחינת זרימת המידע**, אבל **לא** Generator שאומן על Belief ולא Selector שאומן על EIG. משתמשים במודל Generator קיים, במודל Sensor קיים וב־Selector היוריסטי שקוף. אין הבטחה לשיפור AUC.

## מה קורה בסיבוב t

1. Generator Qwen2.5-1.5B + frozen V4.6 LoRA מקבל **Target Role**, תיאור, דירוג/ציוני 20 ה־Roles, התפלגות Marginal היוריסטית עם אילוץ בדיוק 3 תפקידים, **כל השאלות הקודמות של Role המטרה**, ואת 3 התשובות האחרונות (גלויות בלבד). הוא יוצר 8 שאלות *חדשות* (עד 3 ניסיונות לכל משבצת). ההנחיה היא HMHY קצרה, לרוב עד 50 מילים, עצמאית/Stateless.
2. Selector יוריסטי, שאינו מאומן על EIG, משקלל התאמה לתפקיד, הבחנה מתחומים קרובים, חידוש מול שאלות קודמות ואי־ודאות Belief. הוא בוחר **אחת מתוך 8 לכל Role**. אם אין מועמדת בטוחה/עצמאית, עוצרים ולא מבצעים קריאות MAS.
3. מריצים 20 שאילתות בתשלום מול **מערכת MAS5 אחת** (ה־Executor הקיים). Truth/trace נשמרים בקבצים נפרדים, ו־Generator לא קורא אותם.
4. Sensor Qwen2.5-3B + frozen V4 adapter קורא `(role, user query, public final response)` עד 1536 tokens ומחזיר Logit. ממוצע ה־Logits לאורך הסיבובים מעדכן Belief, כולל דירוג 20 Roles ו־1140 תצורות אפשריות בדיוק 3 פעילים. המרג'ינלים **אינם הסתברויות מכוילות**.
5. רק אחרי כתיבת Belief ציבורי מחושב Within-MAS AUC ו־Exact Top-3 בעזרת התוויות הפרטיות. דוח אמת נפרד ולא זמין לקלט ההתקפה. חוזרים לסיבוב הבא עם Belief חדש.

**8 סיבובים × 20 Roles = 160 הרצות MAS בתשלום.** ה־Generator מייצר 8 × 20 × 8 = 1280 מועמדות (לפני ניסיונות יצירה חוזרים) ללא קריאות MAS. ההרצה כוללת 24 עבודות Slurm תלויות. תקציב זמן תכנוני: כמה שעות עד לילה ויותר, תלוי בתור GPU, זמן Sampling וקצב API. כל Job מוגבל ל־20 שעות. עלות ספק ה־API אינה ידועה מראש.

## התקנה על שרת BGU

מהמחשב המקומי לאחר הורדת ה־ZIP:

```bash
scp ~/Downloads/mas5_v58_adaptive_8round.zip wieseld@slurm.bgu.ac.il:~/mas_rl_detective/
```

ב־BGU:

```bash
cd ~/mas_rl_detective
unzip -n mas5_v58_adaptive_8round.zip
python tests/test_v58_cpu.py  # אפשר גם עם Conda Python; ללא מודל וללא API
bash cluster/role_direct_within_v4/v58_submit.sh
RUN=$(ls -dt artifacts/role_direct_within_v4/adaptive_v58/mas5_* | head -1)
cat "$RUN/public_preflight.json"
```

**עד כאן אין עבודות Slurm ואין קריאות API.** ברירת המחדל היא מערכת DEV הראשונה בסדר לקסיקוגרפי. לבחירת מערכת: `V58_SESSION_ID=simple5_... bash .../v58_submit.sh`.

## Preview חינמי לפני התשלום (מומלץ)

```bash
bash cluster/role_direct_within_v4/v58_preview.sh "$RUN"
JOB=$(cat "$RUN/preview_job_id.txt")
sacct -j "$JOB" -X --format=JobID,JobName%24,State,ExitCode,Elapsed
# לאחר COMPLETED:
cat "$RUN/generator_report_round_01.json"
column -s $'\t' -t "$RUN/review_selected_round_01.tsv" | head -25
```

יצירת 160 שאלות מועמדות (8 ל־Role) במודל Frozen על GPU; **ללא קריאות MAS**. זמן צפוי: כ־5–45 דקות עבודה על GPU, בתוספת המתנה בתור (תלוי במספר הניסיונות/קצב המודל). אם הגנרציה נכשלת, בדוק `generator_failure_round_01.json` ואת `pool_partial_dev_round_01.jsonl`; אל תגיש מחדש עבודת תשלום.

## הרצת פיילוט 8 סיבובים — אישור חיוב מפורש

רק אחרי שעברת על שאלות Preview:

```bash
V58_APPROVE_MAS_EXECUTIONS=160 \
  bash cluster/role_direct_within_v4/v58_launch.sh "$RUN"
```

הפקודה מזהה Preview מוכן ומשתמש בו **מבלי לייצר את R1 מחדש**. אם לא הרצת Preview, היא תיצור R1 כחלק מהשרשרת. היא מסרבת להגשה נוספת אם כבר קיים `jobs.tsv`.

## AUC חי

```bash
bash cluster/role_direct_within_v4/v58_watch.sh "$RUN"
watch -n 20 "cat '$RUN/live_auc_history.jsonl' 2>/dev/null || echo 'Waiting for first round...'"
```

AUC מדווח **למערכת אחת בלבד** לאחר כל סיבוב: 20 Role scores, בדיוק 3 חיוביים. אין להשוות אותו ישירות לממוצע AUC של חמש המערכות ב־V5.6. לפירוט המלא: `cat "$RUN/live_auc_latest.json"`.

## מה נבדק ומה לא

- בדיקות CPU מקומיות: חישוב Belief × 8, אילוץ שלושה Roles, קליטת Belief ל־Generator, בחירת מועמדת, חישוב Within-MAS AUC, שמירת דוחות ובידוד נתוני אמת.
- נבדק Bash syntax לכל קובצי ה־Slurm.
- **לא** הורץ אינפרנס Qwen GPU או MAS API ב־BGU במסגרת הכנת החבילה; Preview נועד לחשוף בעיות לפני חיוב.
- ה־Sensor עדיין לא מכויל; Fusion של ממוצע Logits הוא Baseline. ה־Selector אינו ranker מאומן; זו לולאה מותנית־Belief, לא מדיניות EIG מאומנת. אפשר לשנות את ה־Checkpoint ב־`V58_SENSOR_ADAPTER=...` רק בזמן `prepare`; לא משנים באמצע פיילוט.
- ההרצה על MAS5 שכבר שימשו באימון/בבדיקות היא בדיקת היתכנות In-Sample, לא בדיקת הכללה. 
- אין לשנות או לבטל עבודות V5.6 או V5.7; התיקייה וה־Checkpoints החדשים מבודדים.
