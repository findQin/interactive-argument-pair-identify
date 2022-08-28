import argparse


def get_args(data_dir, gpu_ids):

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default = "BertOrigin", type = str, help = "模型的名字")

    parser.add_argument("--data_dir",
                        default = data_dir,
                        type = str)
                        
    parser.add_argument("--bert_model_dir",
                        default = '../work/lbwj_bin/model/bert/bert-base-chinese/',
                        type = str)

    parser.add_argument("--output_dir",
                        default = "../work/lbwj_bin/model/bert/",
                        type = str)

    parser.add_argument("--best_model_file",
                        default = "../work/lbwj_bin/model/bert/model.bin",
                        type = str)

    parser.add_argument("--max_seq_length",
                        default = 512,
                        type = int,
                        help = "The maximum total input sequence length after WordPiece tokenization. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")

    # 训练参数
    parser.add_argument("--batch_size",
                        default = 6,
                        type = int,
                        help = "Total batch size for training.")

    parser.add_argument("--num_train_epochs",
                        default = 10,
                        type = int,
                        help = "Total number of training epochs to perform.")
                        
    parser.add_argument("--warmup_ratio",
                        default = 0.1,
                        type = int,
                        help = "Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")
                        
    # optimizer 参数
    parser.add_argument("--learning_rate",
                        default = 5e-5,
                        type = float,
                        help = "Adam 的 学习率"
                        )
                        
    parser.add_argument("--hidden_dropout_prob",
                        default = 0.01,
                        type = float                       
                        )
                        
    parser.add_argument("--hidden_size",
                        default = 768,
                        type = int)
                        
    parser.add_argument("--max_grad_norm",
                        default = 1e-3,
                        type = float
                        )                    
                                     
    parser.add_argument("--gpu_ids", 
                        default = gpu_ids,
                        type = str,  
                        help = "gpu 的设备id")
    
    config = parser.parse_args()

    return config
