

name = "ceval-5shot_bf16_nf4_gs64.log"
# name = "mmlu_5shot_bf16.log"
with open(name) as f:
    data = f.read().strip().split("\n")

cnt = 0
t = 0

for i in range(len(data)):
    if not data[i][0] == "[":
        continue
    cnt += 1
    t0 = float(data[i].split("] [")[1].split(" ")[0])
    t += t0
    print(t0)

print(t, cnt, t/cnt)

