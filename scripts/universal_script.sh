#!/bin/bash

trap "exit" INT

GPU=0
total_steps=10001
seed=2345
#DATASETS=("co_cs"  "chameleon" "co_photo" "actor" "wiki_cs" "pubmed" "co_computer" "squirrel" "arxiv" ) #"products"  

DATASETS=("actor" "squirrel")
#TASKS=("p_link" "p_recon" "p_ming" "p_decor" "p_minsg")
TASKS=("p_decor")
#MULTI_TASK_STR="p_link p_recon p_ming p_decor p_minsg"

#PARETO_OPTIONS=(" " "--not_use_pareto")

for dataset in "${DATASETS[@]}"
do
#    for pareto_option in "${PARETO_OPTIONS[@]}"
#    do
#        # name is pareto if pareto_option is empty else no_pareto
#        name="pareto"
#        if [ "$pareto_option" == "--not_use_pareto" ]; then
#            name="no_pareto"
#        fi
#        echo "LINK: Running all tasks on ${dataset} with pareto option ${pareto_option}"
#        bash link_${dataset}.sh $GPU $total_steps $seed "$MULTI_TASK_STR" $name $pareto_option 
#        wait
#    done
#
    for task in "${TASKS[@]}"
    do
        name=$task
        echo "LINK: Running ${task} on ${dataset} with no pareto option"
        bash link_${dataset}.sh $GPU $total_steps $seed $task $name "--not_use_pareto" 
        wait
    done
    wait
done

wait

for dataset in "${DATASETS[@]}"
do
    # for pareto_option in "${PARETO_OPTIONS[@]}"
    # do
    #     name="pareto"
    #     if [ "$pareto_option" == "--not_use_pareto" ]; then
    #         name="no_pareto"
    #     fi
    #     echo "SSNC: Running all tasks on ${dataset} with pareto option ${pareto_option}"
    #     bash ssnc_${dataset}.sh $GPU $total_steps $seed "$MULTI_TASK_STR" $name $pareto_option 
    #     wait
    # done

    for task in "${TASKS[@]}"
    do
        name=$task
        echo "SSNC: Running ${task} on ${dataset} with no pareto option"
        bash ssnc_${dataset}.sh $GPU $total_steps $seed $task $name "--not_use_pareto" 
        wait
    done
    wait
done
