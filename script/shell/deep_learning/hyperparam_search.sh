#!/usr/bin/bash
lrs=( 1e-4 )
batch_sizes=( 64 )
embd_models=( "w2v" "glove" "fasttext" )
hidden_dims=( 64 256 )
lstm_layerss=(1 3)
training_ratios=( 1 )
margins=( 0.2 )
preprocessings=( "nltk" "spacy" "stanza")

for lr in "${lrs[@]}"; 
do
    for batch_size in "${batch_sizes[@]}";
    do
        for hidden_dim in "${hidden_dims[@]}";
        do
            for lstm_layers in "${lstm_layerss[@]}";
            do
                for training_ratio in "${training_ratios[@]}";
                do
                    for margin in "${margins[@]}";
                    do
                        for preprocessing in "${preprocessings[@]}";
                        do
                            for embd_model in "${embd_models[@]}";
                            do
                                if [ "$embd_model" = "w2v" ];
                                then
                                    python model.py --lr $lr --batch_size $batch_size --embd_model "$embd_model" --hidden_dim $hidden_dim --lstm_layers $lstm_layers --training_ratio $training_ratio --margin $margin --preprocessing "$preprocessing"
                                elif [ "$embd_model" = "fasttext" ];
                                then
                                    embd_names=( "en" "simple" )
                                    for embd_name in "${embd_names[@]}";
                                    do
                                        python model.py --lr $lr --batch_size $batch_size --embd_model "$embd_model" --hidden_dim $hidden_dim --lstm_layers $lstm_layers --training_ratio $training_ratio --margin $margin --preprocessing "$preprocessing" --embd_name "$embd_name"
                                    done
                                elif [ "$embd_model" = "glove" ];
                                then
                                    embd_names=( "6B" "42B" "840B" )
                                    for embd_name in "${embd_names[@]}";
                                    do
                                        python model.py --lr $lr --batch_size $batch_size --embd_model "$embd_model" --hidden_dim $hidden_dim --lstm_layers $lstm_layers --training_ratio $training_ratio --margin $margin --preprocessing "$preprocessing" --embd_name "$embd_name"
                                    done
                                    embd_dims=( 50 100 200 )
                                    for embd_dim in "${embd_dims[@]}";
                                    do
                                        python model.py --lr $lr --batch_size $batch_size --embd_model "$embd_model" --hidden_dim $hidden_dim --lstm_layers $lstm_layers --training_ratio $training_ratio --margin $margin --preprocessing "$preprocessing" --embd_name "6B" --embd_dim $embd_dim
                                    done
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done