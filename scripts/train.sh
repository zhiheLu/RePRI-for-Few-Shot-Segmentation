DATA=$1
SPLIT=$2
GPU=$3
LAYERS=$4
ARCH=$5
CLASSES=$6
CLASSESVAL=$7
EPOCHS=$8


dirname="results/train/${ARCH}-${LAYERS}/${DATA}/split_${SPLIT}"
mkdir -p -- "$dirname"
python3 -m src.train --config config_files/${DATA}.yaml \
					 --opts train_split ${SPLIT} \
						    layers ${LAYERS} \
						    gpus ${GPU} \
						    num_classes_tr ${CLASSES} \
						    num_classes_val ${CLASSESVAL} \
						    arch ${ARCH} \
						    batch_size 12 \
						    epochs ${EPOCHS} \
 						    visdom_port -1 \
						    episodic_val True \
							    | tee ${dirname}/log_shot1.txt
