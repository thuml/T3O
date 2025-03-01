import os
import time
import numpy as np
import argparse
import random
import torch
import gradio as gr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from io import StringIO
from test_inference import test_inference
from exceptions import MineException
from exp.exp_large_few_shot_roll_demo import Exp_Large_Few_Shot_Roll_Demo
import altair as alt


def load_data(temp_file,progress=gr.Progress()):
    '''
    Descriptions:
        Original data loading entrance to get original data and data columns.
    Inputs: 
        .csv file path -> string format
    Outputs: 
        csv file content -> pd.Dataframe format
        csv file column list -> gr.Dropdown() format
    Remarks:
        I downloaded the input data to "temp_file.csv" during this function for easier data accessing.
    '''
    progress(0, desc="Starting...")
    df = pd.read_csv(temp_file)
    df.to_csv('temp_file.csv', index=False)
    columns = df.columns.tolist()
    plot_target=gr.Dropdown(choices=columns,allow_custom_value=True,interactive=True)
    return plot_target

def load_table():
    df = pd.read_csv("temp_file.csv")
    return df

def plot_column(choice):
    '''
    Descriptions:
        Plot function for target column using Matplotlib. (beta)
    Inputs: 
        choice of target column -> string format
    Outputs: 
        Matplotlib image format
    '''
    df = pd.read_csv("temp_file.csv")
    df_choice = df[choice]
    image=plt.figure()
    plt.title('Value over Time:{}'.format(choice))
    plt.plot(df_choice)
    return image

def plot_column_lineplt(choice):
    '''
    Descriptions:
        Plot function for target column using gr.LinePlot(). (beta)
    Inputs: 
        choice of target column -> string format
    Outputs: 
        gr.LinePlot() input -> pd.DataFrame format
    Remarks:
        Remember to use "alt.data_transformers.disable_max_rows()" to disable the 5000 rows checking mechanism
    '''
    df = pd.read_csv("temp_file.csv")
    df_choice = np.array(df[choice])
    length = len(df_choice)
    alt.data_transformers.disable_max_rows()
    df_new = pd.DataFrame({
        'x' : np.arange(length),
        'y': df_choice[:length]
    })
    return df_new

def plot_column_plotly(choice):
    '''
    Descriptions:
        Plot function for target column using Plotly.
    Inputs: 
        choice of target column -> string format
    Outputs: 
        Plotly image format
    '''
    df = pd.read_csv("temp_file.csv")
    df_choice = np.array(df[choice])
    length = len(df_choice)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(length),y=df_choice,mode='lines',name="Ground Truth"))
    fig.update_layout(title='Value over Time:{}'.format(choice),xaxis_title='Timepoint',yaxis_title='{} value'.format(choice))
    return fig

def inference_matplotlib(choice,start_index,end_index,pred_len):
    '''
    Descriptions:
        Inference function for target column using Matplotlib. (beta)
    Input: 
        choice of target column -> string format
        start_index -> string format
        end_index -> string format
        pred_len -> string format
    Output: 
        result_dir -> string format
        Matplotlib image format
    Remarks:
        patch_len is now fixed to 96, it might be another input in future adjustment
    '''
    start_index = int(start_index)
    end_index = int(end_index)
    pred_len = int(pred_len)
    df = pd.read_csv("temp_file.csv")
    df_choice = df[choice]
    length = len(df_choice)
    test_data = df_choice.to_numpy().reshape((length,1))
    result_dir = test_inference(test_data[start_index:end_index,:],pred_len)
    original_data = np.load(os.path.join(result_dir, 'original_data.npy'))
    pred_data = np.load(os.path.join(result_dir, 'pred_data.npy'))
    seq_len = end_index-start_index
    patch_len=96
    pred_len = pred_len
    image= plt.figure(figsize=((seq_len + pred_len) // patch_len * 5, 5))
    plt.plot(np.arange(seq_len + pred_len), pred_data, label='Prediction', c='dodgerblue', linewidth=2)
    plt.plot(np.arange(seq_len), original_data, label='GroundTruth', c='tomato', linewidth=2)
    plt.legend()
    return result_dir,image

def inference_plotly(choice,start_index,end_index,pred_len,show_GT):
    '''
    Descriptions:
        Inference function for target column using Plotly.
    Input: 
        choice of target column -> string format
        start_index -> string format
        end_index -> string format
        pred_len -> string format
        show_GT -> bool format
    Output: 
        result_dir -> string format
        Matplotlib image format
    '''
    start_index = int(start_index)
    end_index = int(end_index)
    pred_len = int(pred_len)
    df = pd.read_csv("temp_file.csv")
    df_choice = df[choice]
    length = len(df_choice)
    if start_index<0 or start_index>length-1:
        raise MineException("起始位置错误")
    if end_index<0 or end_index>length-1:
        raise MineException("终止位置错误")
    if pred_len<0:
        raise MineException("预测长度错误")
    test_data = df_choice.to_numpy().reshape((length,1))
    result_dir = test_inference(test_data[int(start_index):int(end_index),:],int(pred_len))
    original_data = np.load(os.path.join(result_dir, 'original_data.npy')).squeeze()
    pred_data = np.load(os.path.join(result_dir, 'pred_data.npy')).squeeze()
    seq_len = int(end_index)-int(start_index)
    pred_len = int(pred_len)
    fig = go.Figure()
    if show_GT:
        GT_len = pred_len
        try:
            if end_index + pred_len > length:
                GT_len = length-end_index
                raise MineException("预测范围超出GroundTruth限制")
        except Exception as result:
            GT_len = length-end_index
            print(result)
        groundtruth_data = test_data[end_index-1:end_index+GT_len].squeeze()
        fig.add_trace(go.Scatter(x=np.arange(seq_len + pred_len),y=pred_data,mode='lines',name="Prediction",line = dict(color='royalblue', width=1.5)))
        fig.add_trace(go.Scatter(x=np.arange(seq_len),y=original_data,mode='lines',name="Original Data",line = dict(color='firebrick', width=1.5)))
        if GT_len>0:
            fig.add_trace(go.Scatter(x=np.arange(seq_len-1,seq_len + GT_len),y=groundtruth_data,mode='lines',name="Ground Truth Data",line = dict(color='purple', width=1.5)))
        fig.update_layout(title='Two Line Plots', xaxis_title='Timepoint', yaxis_title='{} value'.format(choice))
    else:
        fig.add_trace(go.Scatter(x=np.arange(seq_len + pred_len),y=pred_data,mode='lines',name="Prediction",line = dict(color='royalblue', width=1.5)))
        fig.add_trace(go.Scatter(x=np.arange(seq_len),y=original_data,mode='lines',name="Original Data",line = dict(color='firebrick', width=1.5)))
        fig.update_layout(title='Two Line Plots', xaxis_title='Timepoint', yaxis_title='{} value'.format(choice))
    return result_dir,fig

def inference_plotly_fast(choice,start_index,end_index,pred_len,show_GT):
    '''
    Descriptions:
        Inference function for target column using Plotly.
    Input: 
        choice of target column -> string format
        start_index -> string format
        end_index -> string format
        pred_len -> string format
        show_GT -> bool format
    Output: 
        result_dir -> string format
        Matplotlib image format
    '''
    start_index = int(start_index)
    end_index = int(end_index)
    pred_len = int(pred_len)
    df = pd.read_csv("temp_file.csv")
    df_choice = df[choice]
    length = len(df_choice)
    if start_index<0 or start_index>length-1:
        raise MineException("起始位置错误")
    if end_index<0 or end_index>length-1:
        raise MineException("终止位置错误")
    if pred_len<0:
        raise MineException("预测长度错误")
    test_data = df_choice.to_numpy().reshape((length,1))
    result_dir = exp.inference(test_data[int(start_index):int(end_index),:],int(pred_len))
    original_data = np.load(os.path.join(result_dir, 'original_data.npy')).squeeze()
    pred_data = np.load(os.path.join(result_dir, 'pred_data.npy')).squeeze()
    seq_len = int(end_index)-int(start_index)
    pred_len = int(pred_len)
    fig = go.Figure()
    if show_GT:
        GT_len = pred_len
        try:
            if end_index + pred_len > length:
                GT_len = length-end_index
                raise MineException("预测范围超出GroundTruth限制")
        except Exception as result:
            GT_len = length-end_index
            print(result)
        groundtruth_data = test_data[end_index-1:end_index+GT_len].squeeze()
        fig.add_trace(go.Scatter(x=np.arange(seq_len + pred_len),y=pred_data,mode='lines',name="Prediction",line = dict(color='royalblue', width=1.5)))
        fig.add_trace(go.Scatter(x=np.arange(seq_len),y=original_data,mode='lines',name="Original Data",line = dict(color='firebrick', width=1.5)))
        if GT_len>0:
            fig.add_trace(go.Scatter(x=np.arange(seq_len-1,seq_len + GT_len),y=groundtruth_data,mode='lines',name="Ground Truth Data",line = dict(color='purple', width=1.5)))
        fig.update_layout(title='Two Line Plots', xaxis_title='Timepoint', yaxis_title='{} value'.format(choice))
    else:
        fig.add_trace(go.Scatter(x=np.arange(seq_len + pred_len),y=pred_data,mode='lines',name="Prediction",line = dict(color='royalblue', width=1.5)))
        fig.add_trace(go.Scatter(x=np.arange(seq_len),y=original_data,mode='lines',name="Original Data",line = dict(color='firebrick', width=1.5)))
        fig.update_layout(title='Two Line Plots', xaxis_title='Timepoint', yaxis_title='{} value'.format(choice))
    return result_dir,fig

# '''
# This is older version -- begin
# '''
# # 文件夹路径
# folder_path = 'test_results/large_finetune_2G_{672}_{96}_{96}__Timer_ETTh1_ftM_sl672_ll576_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp24-04-10_20-04-02/ETTh1.csv/96/'

# # 列出文件夹中的所有 .npz 文件
# file_list = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

# def visual(number, true, preds=None):
#     """
#     Results visualization
#     """
#     fig = plt.figure()
#     if preds is not None:
#         plt.plot(preds, label='Prediction', c='dodgerblue', linewidth=2)
#     plt.plot(true, label='GroundTruth', c='tomato', linewidth=2)
#     plt.title("number {} of {}".format(number,len(file_list)))
#     plt.legend(loc='upper left')
#     return fig


# def draw_all():
#     for i in range(len(file_list)):
#         time.sleep(1)
#         file_path = os.path.join(folder_path,file_list[i])
#         with np.load(file_path) as data:
#             groundtruth = data['groundtruth']
#             predict = data['predict']
#         image = visual(i+1, groundtruth, predict)
#         yield image
# '''
# This is older version -- end
# '''

'''
load checkpoint once
'''
parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, default='large_finetune')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=str, default='Timer')
parser.add_argument('--ckpt_path', type=str, default='checkpoints/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt')

# model define
parser.add_argument('--patch_len', type=int, default=96)
parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
parser.add_argument('--e_layers', type=int, default=8, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true')
# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


args = parser.parse_args()
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
exp = Exp_Large_Few_Shot_Roll_Demo(args)


'''
The Timer Demo App in Gradio
'''
with gr.Blocks() as demo:
    gr.Markdown("<div align='center' ><font size='70'><b>Timer时间序列大模型</b></font></div>") 
    with gr.Tab("接口调用"):
        gr.Markdown("# 数据上传（仅csv格式）")
        gr.Markdown("请在此处上传需要分析的文件，上传后的文件将自动在下面的表格中展示。")
        upload_button = gr.File(label="请在此上传文件")
        table_button = gr.Button(value = "展示表格")
        table = gr.Dataframe(interactive=True)

        gr.Markdown("# 目标变量选择")
        gr.Markdown("上传文件中的所有变量如下，请选择要分析的变量。你可以在图中查看所选变量的变化趋势。")
        plot_target = gr.Dropdown(allow_custom_value=True,interactive=True)
        file = upload_button.upload(fn=load_data, inputs=upload_button, outputs=[plot_target], api_name="upload_csv")
        table_button.click(fn=load_table, inputs=None, outputs=table)

        colplot= gr.Plot()
        plot_target.change(fn=plot_column_plotly, inputs=[plot_target],outputs=colplot)

        gr.Markdown("# 目标变量推理")
        gr.Markdown("请指定目标变量分析范围起始位置与终止位置，并给出想要预测的长度。准备妥当后，点击按钮进行推理。推理结果将在图中展示。")
        with gr.Row():
            start = gr.Textbox(label="start position")
            end = gr.Textbox(label="end position")
            pred = gr.Textbox(label="pred_len")
            show_GT = gr.Checkbox(label="show ground truth")

        
        inference_button = gr.Button(value = "Inference Now")
        result_dir = gr.Textbox(label="result directory")
        result_img = gr.Plot()
        inference_button.click(
            fn=inference_plotly_fast,
            inputs=[plot_target,start,end,pred,show_GT],
            outputs=[result_dir,result_img]
        )

    # with gr.Tab("可视化展示"):
    #     with gr.Column():
    #         with gr.Row():
    #             all = gr.Textbox(len(file_list),label="测试集样本数量")
    #     map = gr.Plot()
    #     btn = gr.Button("RUN")
    #     btn.click(fn=draw_all,inputs=[],outputs=map)
    
demo.queue()
demo.launch(root_path="http://anylearn.nelbds.cn:81/hdemo/timer") # FIXME: hardcoded root path
