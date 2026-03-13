## original generation
export CUDA_VISIBLE_DEVICES=0,1,3
current_time=$(date +"%Y%m%d-%H%M%S")

# replace the following paths before running
model_name=stable-diffusion-v1-4
data_path=<--path-to-your-original-data-->
disturb_1_data_path=<--path-to-your-token-view-perturbation-->
disturb_2_data_path=<--path-to-your-style-view-perturbation-->
disturb_3_data_path=<--path-to-your-semantic-view-perturbation-->
output_root=res/${current_time}
clip_model=clip-vit-large-patch14
blip_model_name=blip2-opt-6.7b
inference_batch_size=90

mkdir -p ${output_root}

## disturb caption 1
# generate images
python src/reconstruct.py \
    --pretrained_model_name_or_path ${model_name} \
    --data_dir ${disturb_1_data_path} \
    --num_validation_images 10 \
    --save_dir ${output_root}/disturb_caption_1 \
    --batch_size ${inference_batch_size} \
    --inference 40 \
    --resume

## disturb caption 2
# generate images
python src/reconstruct.py \
    --pretrained_model_name_or_path ${model_name} \
    --data_dir ${disturb_2_data_path} \
    --num_validation_images 10 \
    --save_dir ${output_root}/disturb_caption_2 \
    --batch_size ${inference_batch_size} \
    --inference 40 \
    --resume

## disturb caption 3
# generate images
python src/reconstruct.py \
    --pretrained_model_name_or_path ${model_name} \
    --data_dir ${disturb_3_data_path} \
    --num_validation_images 10 \
    --save_dir ${output_root}/disturb_caption_3 \
    --batch_size ${inference_batch_size} \
    --inference 40 \
    --resume


## original image embedding
# generate caption
python src/caption_generate.py \
    --data_path ${data_path} \
    --blip_model_name ${blip_model_name} \
    --output_path ${output_root}/target/caption_blip2_opt_6.7b.json \
    --batch_size 32
# calculate embedding
python src/cal_embedding.py \
    --data_dir ${output_root}/target/caption_blip2_opt_6.7b.json \
    --output_dir ${output_root}/target/embedding_blip2_opt_6.7b_ \
    --gpu 0 \
    --image_encoder clip \
    --clip_model ${clip_model} \
    --conditioning concate \

## disturb caption 1
# generate caption
python src/caption_generate.py \
    --data_path ${output_root}/disturb_caption_1/metafile.json \
    --blip_model_name ${blip_model_name} \
    --output_path ${output_root}/disturb_caption_1/caption_blip2_opt_6.7b.json \
    --batch_size 32

# calculate embedding
python src/cal_embedding.py \
    --data_dir ${output_root}/disturb_caption_1/caption_blip2_opt_6.7b.json \
    --output_dir ${output_root}/disturb_caption_1/embedding_blip2_opt_6.7b \
    --gpu 0 \
    --image_encoder clip \
    --clip_model ${clip_model} \
    --conditioning concate \

# calculate similarity
python src/cal_relevance.py \
    --target_embeddings ${output_root}/target/embedding_blip2_opt_6.7b_/embeddings.json \
    --disturbed_caption_gen_embeddings ${output_root}/disturb_caption_1/embedding_blip2_opt_6.7b/embeddings.json \

## disturb caption 2
# generate caption
python src/caption_generate.py \
    --data_path ${output_root}/disturb_caption_2/metafile.json \
    --blip_model_name ${blip_model_name} \
    --output_path ${output_root}/disturb_caption_2/caption_blip2_opt_6.7b.json \
    --batch_size 32

# calculate embedding
python src/cal_embedding.py \
    --data_dir ${output_root}/disturb_caption_2/caption_blip2_opt_6.7b.json \
    --output_dir ${output_root}/disturb_caption_2/embedding_blip2_opt_6.7b \
    --gpu 0 \
    --image_encoder clip \
    --clip_model ${clip_model} \
    --conditioning concate \

# calculate similarity
python src/cal_relevance.py \
    --target_embeddings ${output_root}/target/embedding_blip2_opt_6.7b_/embeddings.json \
    --disturbed_caption_gen_embeddings ${output_root}/disturb_caption_2/embedding_blip2_opt_6.7b/embeddings.json \

## disturb caption 3
# generate caption
python src/caption_generate.py \
    --data_path ${output_root}/disturb_caption_3/metafile.json \
    --blip_model_name ${blip_model_name} \
    --output_path ${output_root}/disturb_caption_3/caption_blip2_opt_6.7b.json \
    --batch_size 32

# calculate embedding
python src/cal_embedding.py \
    --data_dir ${output_root}/disturb_caption_3/caption_blip2_opt_6.7b.json \
    --output_dir ${output_root}/disturb_caption_3/embedding_blip2_opt_6.7b \
    --gpu 0 \
    --image_encoder clip \
    --clip_model ${clip_model} \
    --conditioning concate \

# calculate similarity
python src/cal_relevance.py \
    --target_embeddings ${output_root}/target/embedding_blip2_opt_6.7b_/embeddings.json \
    --disturbed_caption_gen_embeddings ${output_root}/disturb_caption_3/embedding_blip2_opt_6.7b/embeddings.json \