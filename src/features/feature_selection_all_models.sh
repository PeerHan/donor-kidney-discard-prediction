source .env

declare -a arr=("DT" "LR" "RF" "XGB" "MLP")
declare -a berr=("paper")


for model in "${arr[@]}"
do
for featureset in "${berr[@]}"
do

   echo "Model $model on $featureset"
   trainpath="$FEATURE_DIR/$featureset/train.csv"
   echo $trainpath

   python -m src.features.create_paper_feature_set \
   -m "$model" -t 1 \
   -p "$MODEL_RESULTS" \
   -tf $trainpath \
   -j -1 -pen 0.0005

done
done