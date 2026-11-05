source .env

declare -a arr=("LR" "DT" "XGB" "RF")
declare -a featuresets=("FS")

for fs in "${featuresets[@]}"
do
   mkdir "$MODEL_RESULTS/$fs"
   for model in "${arr[@]}"
   do
      echo "Model $model on $fs"
      mkdir "$MODEL_RESULTS/$fs/$model"
      trainpath="$FEATURE_DIR/$([[ $fs == *"exploration"* || $fs == *"FS"* ]] && echo "$fs/$model/train.csv" || echo "$fs/train.csv")"
      testpath="$FEATURE_DIR/$([[ $fs == *"exploration"* || $fs == *"FS"* ]] && echo "$fs/$model/test.csv" || echo "$fs/test.csv")"
      echo $trainpath
      echo $testpath

      python -m src.modeling.tune_and_eval_model \
      -m "$model" -t 100 -p "$MODEL_RESULTS/$fs/$model"\
      -tf $trainpath -y "Transplantationsstatus" -tsf $testpath\
      -i 1 -d "cpu" -s True -fs $fs
   done
done