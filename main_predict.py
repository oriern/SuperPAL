from utils import *
from docSum2MRPC_Aligner import docSum2MRPC_Aligner
sys.path.append('/transformers/examples/')

import run_glue

import contextlib
@contextlib.contextmanager
def redirect_argv(num):
    sys._argv = sys.argv[:]
    sys.argv = str(num).split()
    yield
    sys.argv = sys._argv






parser = argparse.ArgumentParser()
parser.add_argument('-data_path', type=str, default='/home/nlp/ernstor1/DUC2004/')  # 'data/final_data/data')
parser.add_argument('-mode', type=str, default='dev')
parser.add_argument('-log_file', type=str, default='results/dev_log.txt')
parser.add_argument('-output_path', type=str, required=True)
parser.add_argument('-alignment_model_path', type=str, required=True)
parser.add_argument('-database', type=str, default='None')
args = parser.parse_args()




aligner = docSum2MRPC_Aligner(data_path=args.data_path, mode=args.mode,
                 log_file=args.log_file, output_file = args.output_path,
                 database=args.database)
logging.info(f'output_file_name: {args.output_file}')

summary_files = glob.glob(f"{args.data_path}/summaries/*")
for sfile in summary_files:
        print ('Starting with summary {}'.format(sfile))
        aligner.read_and_split(args.database, sfile)
        aligner.scu_span_aligner()
aligner.save_predictions()
with redirect_argv('python --model_type roberta --model_name_or_path roberta-large-mnli --task_name MRPC --do_eval'
                           f' --calc_final_alignments --weight_decay 0.1 --data_dir {args.output_path}'
                           ' --max_seq_length 128 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 2e-6'
                           ' --logging_steps 500 --num_train_epochs 2.0 --evaluate_during_training  --overwrite_cache'
                           f' --output_dir {args.alignment_model_path}'):
        run_glue.main()





