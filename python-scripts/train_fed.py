import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import lightning.pytorch as pl
import hydra
from hydra_zen import instantiate 
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
import hashlib
import torch
from collections import OrderedDict
import wandb
import numpy as np
from torchmetrics.functional import auroc
from src.xai import IntegratedGradientsExplainer
import pickle

def compute_aggr_aurocs(pl_module, log_prefix='val'):
    aggr_metrics = {}
    for dl_idx, labels_and_logits in pl_module.all_steps_labels_and_logits.items():
        additional_inputs = pl_module.all_steps_additional_inputs[dl_idx] # example: [[('2590_20', '165_20')]] # (batch size 2)
        ids = [additional_input[0] for additional_input in additional_inputs] # asumo que el id es el primer "additional input"
        ids = [item for t in ids for item in t] # now the ids are flattened but its still a list

        all_y_hat = torch.cat([x['y_hat'] for x in labels_and_logits]) # this flattens the batches
        all_y = torch.cat([x['y'] for x in labels_and_logits]) # this flattens the batches        

        if type(ids[0]) == torch.Tensor:
            ids = [str(x.item()) for x in ids]
        unique_ids = set(ids)
        aggregations_max = []
        labels = []
        for unique_id in unique_ids: 
            idcs = list(map(lambda x: x == unique_id, ids))
            if type(idcs[0]) == torch.Tensor:
                idcs = [idx.item() for idx in idcs]
            logits = all_y_hat[idcs]
            aggregations_max.append( logits.max(dim=0)[0][1] )
            labels.append(all_y[idcs][0])
        
        labels = torch.stack(labels)
        auroc_max = auroc(torch.stack(aggregations_max), labels, task='binary')
        aggr_metrics['{}_aggr_auroc/{}'.format(log_prefix, dl_idx)] = auroc_max
    return aggr_metrics

def average_lists(*lists):
    return [sum(values) / len(values) for values in zip(*lists)]

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def generate_id(s):
    hash_obj = hashlib.sha256(s.encode())
    return hash_obj.hexdigest()[:8]

@hydra.main(config_path=None)
def task(cfg):
    overrides = HydraConfig.get().overrides.task
    overrides_str = ''.join(overrides)
    job_type_id = generate_id(overrides_str)
    with open_dict(cfg):
        cfg.hparams_combination = job_type_id
    wandb.init(project=cfg.wandb_project, group=cfg.wandb_group, name=job_type_id)
               
    random_seed = 42
    pl.seed_everything(random_seed)

    trainers_list = []
    lit_modules_list = []
    train_dls_list = instantiate(cfg.dataloaders.train)
    train_dls_list = [x[1] for x in train_dls_list.items()]
    val_dls_list = instantiate(cfg.dataloaders.val)
    val_dls_list = [x[1] for x in val_dls_list.items()]
    test_dls_list = instantiate(cfg.dataloaders.test)
    test_dls_list = [x[1] for x in test_dls_list.items()]

    auc_weights =cfg.get('auc_weights', [1./len(val_dls_list)]*len(val_dls_list))
    auc_weights = np.array(auc_weights)

    m0_trainer = instantiate(cfg.trainer_0)
    m0 = instantiate(cfg.module_0)
    m0_params = get_parameters(m0.model)

    lit_modules_list = [instantiate(cfg.module_0) for n in range(len(train_dls_list))]
    trainers_list = [instantiate(cfg.trainer_0) for n in range(len(train_dls_list))]
 
    for lit_module in lit_modules_list:
        set_parameters(lit_module.model, m0_params)

    best_val_metric = 0.
    for round in range(cfg.num_rounds):

        for module, trainer, train_dataloader in zip(lit_modules_list, trainers_list, train_dls_list):
            trainer.fit(module, train_dataloader)
            trainer.fit_loop.max_epochs += cfg.epochs_per_round
        
        params_list = []
        for module in lit_modules_list:
            params_list.append(get_parameters(module.model))
        params_avg = average_lists(*params_list)
        set_parameters(m0.model, params_avg)
        val_res = m0_trainer.validate(m0, val_dls_list)
        val_metrics_dict = compute_aggr_aurocs(m0, 'val')
        val_metrics_dict = {k: v.cpu() for k,v in val_metrics_dict.items()}
        avg_val_auroc = (np.array(list(val_metrics_dict.values())) * auc_weights).sum()
        val_metrics_dict['avg_val_auroc'] = avg_val_auroc
        if avg_val_auroc > best_val_metric:
            best_val_metric = avg_val_auroc
            m0_trainer.save_checkpoint("best_federated_model.ckpt")

        wandb.log(val_metrics_dict)
        
        for module in lit_modules_list:
            set_parameters(module.model, params_avg)
        
    test_res = m0_trainer.test(m0, test_dls_list, ckpt_path='best_federated_model.ckpt')
    test_metrics_dict = compute_aggr_aurocs(m0, 'test')
    test_metrics_dict = {k: v.cpu() for k,v in test_metrics_dict.items()}

    test_metrics_dict['best_avg_val_auroc'] =  best_val_metric
    wandb.log(test_metrics_dict)
    wandb.finish()

    if cfg.get('make_explanations', False):
        explanations_dict = {}
        ig_explainer = IntegratedGradientsExplainer(m0.model.eval().cuda())
        for test_dl_idx, test_dl in enumerate(test_dls_list):
            explanations = []
            for batch in iter(test_dl):
                explanations.append(ig_explainer.make_explanation(batch[0].cuda(), c=1, internal_batch_size=10))
                torch.cuda.empty_cache()
            explanations_dict[test_dl_idx] = explanations

        with open('explanations.pkl', 'wb') as h:
            pickle.dump(explanations_dict, h)
    return 0

if __name__ == '__main__':
    task()


