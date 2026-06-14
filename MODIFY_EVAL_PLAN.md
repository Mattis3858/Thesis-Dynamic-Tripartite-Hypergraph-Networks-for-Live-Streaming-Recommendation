Evaluation 改動計畫(只動 eval,不動 model / train / 不重訓)
0. 總體目標與不可違反的原則
目前的測試協定太簡單(固定 user、只均勻隨機替換 streamer/room),導致指標全面飽和(HR 多為 1.0000)、ablation 量不出差異、不同任務設定數字完全相同。本次改動要設計一套更難、對所有模型完全固定、可重現的測試協定,並讓所有模型(我的 HT-Transformer 與全部 baseline)及所有 ablation variant 都用同一份協定重新評分。
四條硬原則,任何改動都不可違反:

本次不重訓任何模型。 全部用既有 checkpoint,只重新跑 evaluation 階段。
測試負樣本必須對所有模型完全相同。 同一個 test query 的那 N 個負樣本,必須事先生成一次、存成檔、所有模型讀同一份。嚴禁在評分時即時、依模型分數動態挑負樣本(那會讓比較不公平)。
不得有時間洩漏。 負樣本與正樣本共用同一個 timestamp t,鄰居取樣一律嚴格早於 t,沿用現有 forward 的時間邏輯,不要動。
可重現。 負樣本生成、任何隨機性都用固定 seed,並把生成結果落地成檔案。


1. 把負樣本生成從「評分時即時抽」改成「事前生成、落地存檔、共用」
現況(請先確認): 目前 evaluate_tripartite_ranking() 與 metrics.py 在評分當下用 eval_rng.choice 即時抽負樣本。問題是每次跑、每個模型可能抽到不同負樣本,無法保證跨模型一致。
改成:
新增一個獨立步驟 build_eval_candidates,在所有模型評估之前只跑一次,產生一份固定的候選集檔案(例如 eval_candidates/{split}_{setting}_seed{S}.parquet 或 .npz)。每個 test 正樣本對應一筆紀錄,內容包含:

正樣本 (u, v, w, t)
該筆的 N 個負樣本清單(每個負樣本是一個被替換後的三元組,例如 (u, v'_j, w'_j, t),j = 1..N)
一個 query id 方便對齊

評分階段(evaluate_tripartite_ranking)改成讀這份檔,不再自己抽。所有模型、所有 ablation 一律讀同一份。
驗收: 跑兩個不同模型,印出它們各自第 0、第 100 筆 query 的負樣本清單,必須逐一相同。

2. 設計更難的負採樣分布(取代 uniform 隨機)
均勻隨機抽 (v', w') 太好分辨,模型只要背「這個 user 歷史碰過哪些組合」就能贏。改成結構化負採樣,讓負樣本長得更像正樣本。請實作下列三種負樣本來源,並可用比例混合(預設比例見下):

Popularity-based(50%): 從「該 query 時間 t 附近的時間窗內,全體最熱門的 (v, w) 組合」抽。熱門度用截至 t 之前的歷史互動次數計算(嚴禁用 t 之後的資料,避免洩漏)。這逼模型不能只靠「熱門 vs 冷門」區分。
User-history-based hard(30%): 從「和當前 user 行為相似的其他 user 常互動、但當前 user 在 t 之前沒互動過的 (v, w)」抽。相似度可先用簡單方式(例如共同互動過的 streamer 重疊度)。這逼模型不能只靠「這個 user 碰過沒」這個捷徑。
Uniform(20%): 保留少量均勻隨機,維持基本覆蓋面。

重要 false-negative 防護: 抽負樣本時,必須排除「該 user 在整個資料集(含未來)中其實有互動過的 (v, w)」,避免把真正會發生的互動誤標成負樣本。也就是負樣本要對照 user 的「全時段正樣本集合」做過濾。
比例與時間窗大小設成可調參數(放 config),預設 popularity:hard:uniform = 50:30:20,時間窗先設「t 之前最近的固定筆數或固定時長」,讓 Claude Code 挑一個合理預設並標 TODO 讓我調。

3. 處理「只有 10 個 item / room」的硬限制——調整各 setting 的 N 與取捨
現況問題: 標準協定是 1 正 + 99 負。但若某個 setting 只替換 item 而 item pool 只有 10 個,根本湊不出 99 個負樣本(去掉正樣本最多 9 個),這個 setting 在現有資料上不可行。
請這樣處理:

主 setting(替換 (v, w) 組合): streamer pool 夠大((v,w) 組合空間大),維持 N = 99,沒問題。這是論文主表用的。
User–Item setting(只替換 item): 因 item pool 過小,直接停用此 setting,不要再產生它的數字(目前它和主 setting 數字完全相同,留著只會招致質疑)。在程式裡保留開關但預設關閉,並印出明確警告說明原因。
User–Streamer setting(只替換 streamer): streamer 有 294 個,可行,維持 N = 99,保留。

把 N 與「啟用哪些 setting」做成 config 參數。

4. 確認指標計算正確,並對齊論文要報告的 K
請逐一確認 / 修正:

NDCG / HR 的定義在 1 正 + N 負的 ranking 下是否正確。 只有 1 個相關項,IDCG = 1,NDCG@K = 1 / log2(rank + 1)(若 rank 從 1 起算)或對應公式;HR@K = 正樣本是否排進前 K。請印出單筆 query 的中間量(rank、各 K 的 HR/NDCG)做 sanity check。
K 的對齊: 目前訓練主流程預設 K = (5,10,20,50,100),但論文表格腳本可能另指定更嚴格的 top-K(如 N@3、N@5)。請確認論文要報告的 K,並讓主評估流程印出的 K 與論文表一致,避免內文宣稱與實際跑的 K 不符。把報告用的 K 集中成一個 config 常數。
同時保留 AUC / AP。 但要清楚標示:在 1 正 + N 負的 ranking 設定下,主指標是 HR@K / NDCG@K,AUC/AP 為輔。


5. 修掉 baseline 評估的明顯 bug,讓比較表可信
用新協定重跑後,請特別檢查下列已知異常(可能是 adaptation / 評估流程的 bug,不是模型本身爛):

HAN 出現「AUC ≈ 0.5(隨機)但 HR/NDCG 全 1.0(完美)」這種數學上自相矛盾的結果。 這必定是評估流程或輸出對接的 bug。請定位:HAN 的輸出分數有沒有正確接進 ranking pipeline?有沒有 NaN 被當成最高分排到第一?修正後重跑。
TGN / LightGCN / HyperHawkes 出現 AUC < 0.5(比隨機還差)。 TGN 是公認強模型,跑出低於隨機高度可疑,懷疑 clique-expansion 的 adaptation 把輸入弄壞。請檢查這幾個 baseline 的圖建構與 forward 輸入是否正確,至少要能解釋為何低於 0.5。
所有 baseline 都必須走第 1 節那份共用的固定候選集,不得各自抽負樣本。


6. 用新協定重跑的範圍(全部用既有 checkpoint,不重訓)
請用既有 checkpoint,在新協定下重新產生以下所有結果:

主比較表(全部 baseline + HT-Transformer):AUC / AP / HR@K / NDCG@K
Bias-aware gate ablation(Full vs w/o gate)
Heterogeneity ablation(Full / w/o type-aware init / w/o heterogeneous co-occurrence)
Group-wise(streamer-biased / item-biased / mixed / new-unknown 四組)
Cold-start(New u / New s / New w 三情境)—— 同樣套新協定的負採樣與 false-negative 防護

每個數字都要報 mean ± std,跨多個 seed。 這次的 seed 指的是「負樣本生成的 seed」與「既有的多個訓練 seed checkpoint」。請把每個 setting 重複 ≥ 3 次(用不同的候選集 seed),報告平均與標準差。這是擋掉「千分之幾的差異是否顯著」質疑的關鍵。

7. 不在本次範圍、但請預留接口(下一階段才做,需重訓)
以下這次先不做,但請把程式寫成容易接上的形式,並留 TODO:

early stopping 改用 ranking 指標(如 NDCG@5)選 checkpoint —— 這需要重訓,屬於下一階段。先把「用 ranking 指標監控 validation」的函式寫好、用開關控制,預設仍維持現狀(BCE),只留接口。
train loss 改 hard negative —— 同樣下一階段,先不動。


8. 交付與驗收清單
請完成後提供:

一份 build_eval_candidates 腳本 + 產生的候選集檔案,附「兩個不同模型讀到的負樣本完全相同」的驗證輸出。
新舊協定的對照:同一個模型(先用 HT-Transformer)在舊協定與新協定下的指標並排,讓我一眼看出任務是否真的變難(HR 是否從 1.0000 掉下來、是否拉開差距)。
全部模型在新協定下的主比較表(mean ± std)。
HAN 自相矛盾、TGN/LightGCN/HyperHawkes < 0.5 的修正說明(改了什麼、修正後數字)。
一段簡短 log 說明:user-item setting 已停用(原因:item pool = 10)。