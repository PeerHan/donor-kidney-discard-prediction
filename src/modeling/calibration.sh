source .env

declare -a arr=("DT" "LR" "RF" "XGB" "MLP")
declare -a berr=("paper-free")

for model in "${arr[@]}"
do
for featureset in "${berr[@]}"
do
   echo "Model $model on $featureset"
   trainpath="$FEATURE_DIR/$([[ $featureset == *"paper"* ]] && echo "$featureset/$model/train.csv" || echo "$featureset/train.csv")"
   testpath="$FEATURE_DIR/$([[ $featureset == *"paper"* ]] && echo "$featureset/$model/test.csv" || echo "$featureset/test.csv")"
   echo $trainpath
   echo $testpath

   python3 -m src.modeling.create_calibration_and_plots \
   -m "$model" -p "$MODEL_RESULTS/$featureset/$model"\
   -tf $trainpath -tsf $testpath
done
done