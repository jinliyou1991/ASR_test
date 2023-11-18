# Update informatin (2023/11/28)

1.  針對 VCTK 及 TMHINTQI dataset調整 分別對應不同語言
   
2. ASR的語言變成強制設定 不使用whisper的自動偵測功能，因為在中文短句中，容易被誤判成他國語言
   
3. 修整WER CER的算法
  3a. 累計ASR的輸出後再用累計的結果計算，而非使用每句的WER、CER做平均
  3b. 保持每句計算WER、CER的功能
  3c. 紀錄輸出每句ASR的結果
  3d. 更新 ASR model 清單 (google, whisper base, whisper large-v2, whisper large-v3)
  3e. 要使用 whisper large-v3 請先更新whisper安裝包到1106的版本，可以使用這個方法更新
      pip install -U openai-whisper
