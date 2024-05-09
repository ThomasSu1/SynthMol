import numpy as np
import logging
import pandas as pd
import pickle as pkl
import random
import os
from threading import Thread, Lock
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    jaccard_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from rdkit import Chem
from rdkit.Chem import AllChem
import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.utils import clip_grad_norm_

from unicore.modules import init_bert_params
from unicore.data import (
    Dictionary, NestedDictionaryDataset, TokenizeDataset, PrependTokenDataset,
    AppendTokenDataset, FromNumpyDataset, RightPadDataset, RightPadDataset2D,
    RawArrayDataset, RawLabelDataset,
)
from unimol.data import (
    KeyDataset, ConformerSampleDataset, AtomTypeDataset,
    RemoveHydrogenDataset, CroppingDataset, NormalizeDataset,
    DistanceDataset, EdgeTypeDataset, RightPadDatasetCoord,
)
from unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol.models.unimol import NonLinearHead, GaussianLayer

from graph.GAT_Layers import GAT
from graph.getFeatures import save_smiles_dicts, get_smiles_array

from fingerprint.fingerprint import FP


logging.basicConfig(filename="model_training.log", level=logging.INFO)
data_path = "./dataset.csv"  


def set_random_seed(random_seed=1024):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    tc.manual_seed(random_seed)
    tc.cuda.manual_seed(random_seed)
    tc.cuda.manual_seed_all(random_seed)  
    tc.backends.cudnn.benchmark = False
    tc.backends.cudnn.deterministic = True
    tc.backends.cudnn.enabled = False


def calculate_molecule_3D_structure():
    def get_smiles_list_():
        data_df = pd.read_csv(data_path)
        smiles_list = data_df["smiles"].tolist()
        smiles_list = list(set(smiles_list))
        print(len(smiles_list))
        return smiles_list

    def calculate_molecule_3D_structure_(smiles_list):
        n = len(smiles_list)
        global p
        index = 0
        while True:
            mutex.acquire()
            if p >= n:
                mutex.release()
                break
            index = p
            p += 1
            mutex.release()

            smiles = smiles_list[index]
            print(index, ':', round(index / n * 100, 2), '%', smiles)

            molecule = Chem.MolFromSmiles(smiles)
            molecule = AllChem.AddHs(molecule)
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coordinate_list = []
            result = AllChem.EmbedMolecule(molecule, randomSeed=42, useRandomCoords=True, maxAttempts=1000)
            if result != 0:
                print('EmbedMolecule failed', result, smiles)
                mutex.acquire()
                with open('./result/invalid_smiles.txt', 'a') as f:
                    f.write('EmbedMolecule failed' + ' ' + str(result) + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                print('MMFFOptimizeMolecule error', smiles)
                mutex.acquire()
                with open('./result/invalid_smiles.txt', 'a') as f:
                    f.write('MMFFOptimizeMolecule error' + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            coordinates = molecule.GetConformer().GetPositions()

            assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
            coordinate_list.append(coordinates.astype(np.float32))

            global smiles_to_conformation_dict
            mutex.acquire()
            smiles_to_conformation_dict[smiles] = {'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
            mutex.release()

    mutex = Lock()
    os.system('rm ./result/invalid_smiles.txt')
    smiles_list = get_smiles_list_()
    global smiles_to_conformation_dict
    smiles_to_conformation_dict = {}
    global p
    p = 0
    thread_count = 16
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_molecule_3D_structure_, args=(smiles_list,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pkl.dump(smiles_to_conformation_dict,
             open('./intermediate/smiles_to_conformation_dict.pkl', 'wb'))
    print('Valid smiles count:', len(smiles_to_conformation_dict))


def construct_data_list():
    data_df = pd.read_csv(data_path)
    smiles_to_conformation_dict = pkl.load(
        open('./intermediate/smiles_to_conformation_dict.pkl', 'rb'))
    data_list = []
    for index, row in data_df.iterrows():
        smiles = row["smiles"]
        if smiles in smiles_to_conformation_dict:
            data_item = {
                "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                "smiles": smiles,
                "label": row["label"],
                "dataset_type": row["dataset_type"],
            }
            data_list.append(data_item)
    pkl.dump(data_list, open('./intermediate/data_list.pkl', 'wb'))


def convert_data_list_to_data_loader():
    def convert_data_list_to_dataset_(data_list):
        dictionary = Dictionary.load('token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        smiles_dataset = KeyDataset(data_list, "smiles")
        label_dataset = KeyDataset(data_list, "label")
        dataset = ConformerSampleDataset(data_list, 1024, "atoms", "coordinates")
        dataset = AtomTypeDataset(data_list, dataset)
        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, False)
        dataset = CroppingDataset(dataset, 1, "atoms", "coordinates", 256)
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=512)
        coord_dataset = KeyDataset(dataset, "coordinates")
        src_dataset = AppendTokenDataset(PrependTokenDataset(token_dataset, dictionary.bos()), dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = AppendTokenDataset(PrependTokenDataset(coord_dataset, 0.0), 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        return NestedDictionaryDataset({
            "input": {
                "src_tokens": RightPadDataset(src_dataset, pad_idx=dictionary.pad(), ),
                "src_coord": RightPadDatasetCoord(coord_dataset, pad_idx=0, ),
                "src_distance": RightPadDataset2D(distance_dataset, pad_idx=0, ),
                "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0, ),
                "smiles": RawArrayDataset(smiles_dataset),
                
            },
            "target": {
                "label": RawLabelDataset(label_dataset),
            }
        })

    batch_size = 128
    data_list = pkl.load(open('./intermediate/data_list.pkl', 'rb'))
    data_list_train = [data_item for data_item in data_list if data_item["dataset_type"] == "train"]
    data_list_validate = [data_item for data_item in data_list if data_item["dataset_type"] == "valid"]
    data_list_test = [data_item for data_item in data_list if data_item["dataset_type"] == "test"]
    dataset_train = convert_data_list_to_dataset_(data_list_train)
    dataset_validate = convert_data_list_to_dataset_(data_list_validate)
    dataset_test = convert_data_list_to_dataset_(data_list_test)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                   collate_fn=dataset_train.collater)
    data_loader_valid = DataLoader(dataset_validate, batch_size=batch_size, shuffle=True,
                                   collate_fn=dataset_validate.collater)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=dataset_test.collater)
    return data_loader_train, data_loader_valid, data_loader_test


class UniMolModel(nn.Module):
    def __init__(self):
        super().__init__()
        dictionary = Dictionary.load('token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), 512, self.padding_idx)
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=15,
            embed_dim=512,
            ffn_embed_dim=2048,
            attention_heads=64,
            emb_dropout=0.1,
            dropout=0.2,
            attention_dropout=0.1,
            activation_dropout=0.0,
            max_seq_len=512,
            activation_fn='gelu',
            no_final_head_layer_norm=True,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, 64, 'gelu'
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        self.apply(init_bert_params)

    def forward(self, sample,):
        net_input = sample['input']
        src_tokens, src_distance, src_coord, src_edge_type = net_input['src_tokens'], net_input['src_distance'], \
                                                             net_input['src_coord'], net_input['src_edge_type']
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        output = {
            "molecule_embedding": encoder_rep,
            "molecule_representation": encoder_rep[:, 0, :],  # get cls token
            "smiles": sample['input']["smiles"],
        }
        return output

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")


class SynthMolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = device
        self.molecule_encoder = UniMolModel()
        self.molecule_encoder.load_state_dict(tc.load('mol_pre_no_h_220816.pt')['model'], strict=False)
        # 存储参数
        self.radius = 6
        self.T = 4
        self.fingerprint_dim = 215
        self.p_dropout = 0.2
        self.gat_encoder = None  

        self.fingerprint_encoder = FP() 

        self.mlp = nn.Sequential(
            nn.Linear(2560,512),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def move_data_batch_to_cuda(self, data_batch):
        data_batch['input'] = {k: v.cuda() if isinstance(v, tc.Tensor) else v for k, v in data_batch['input'].items()}
        data_batch['target'] = {k: v.cuda() if isinstance(v, tc.Tensor) else v for k, v in data_batch['target'].items()}
        return data_batch

    def log_embedding_shapes(
        self,
        molecule_embedding,
        graph_embedding,
        fingerprint_embedding,
        log_file="embedding_shapes.log",
    ):
        with open(log_file, "a") as f:
            f.write(f"Molecule Embedding Shape: {molecule_embedding.shape}\n")
            f.write(f"Graph Embedding Shape: {graph_embedding.shape}\n")
            f.write(f"Fingerprint Embedding Shape: {fingerprint_embedding.shape}\n")

    def forward(self, data_batch):
        data_batch1 = data_batch
        data_batch = self.move_data_batch_to_cuda(data_batch)
        file_path = "./result"
        file_name_prefix = "temp"
        def generateGeoFeatureFile(file_path):
            raw_filename = file_path
            feature_filename = raw_filename.replace(".csv", ".pkl")
            filename = raw_filename.replace(".csv", "")
            smiles_tasks_df = pd.read_csv(raw_filename, index_col=False)
            smiles_tasks_df.sample().reset_index(drop=True)
            smilesList = smiles_tasks_df.smiles.tolist()

            print("number of {} smiles {}: ".format(filename, len(smilesList)))

            if os.path.isfile(feature_filename):
                print("loading*****************************************")
                feature_dicts = pkl.load(open(feature_filename, "rb"))
                # print(feature_dicts)
            else:
                print("generating*****************************************")
                feature_dicts = save_smiles_dicts(smilesList, filename)
            return smiles_tasks_df, feature_dicts

        def prepare_data_from_batch(data_batch1, file_path, file_name_prefix):
            smiles = data_batch1["input"]["smiles"]
            label = data_batch1["target"]["label"]

            if isinstance(smiles, tc.Tensor):
                smiles = smiles.cpu()
            if isinstance(label, tc.Tensor):
                label = label.cpu()

            data_df = pd.DataFrame(
                {
                    "smiles": smiles,
                    "label": label,
                }
            )
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            file_name = file_name_prefix + ".csv"
            csv_file_path = os.path.join(file_path, file_name)

            data_df.to_csv(csv_file_path, index=False)

            return csv_file_path

        molecule_encoder_output = self.molecule_encoder(data_batch)
        molecule_embedding = molecule_encoder_output["molecule_embedding"]

        csv_file_path = prepare_data_from_batch(
            data_batch1, file_path, file_name_prefix
        )
        data_gat, data_gat_feature_dicts = generateGeoFeatureFile(csv_file_path)
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask,smiles_to_rdkit_list = (
            get_smiles_array([data_gat.smiles.values[0]], data_gat_feature_dicts)
        )
        x_atom = tc.Tensor(x_atom).to(self.device)
        x_bonds = tc.Tensor(x_bonds).to(self.device)
        x_atom_index = tc.cuda.LongTensor(x_atom_index).to(self.device)
        x_bond_index = tc.cuda.LongTensor(x_bond_index).to(self.device)
        x_mask = tc.Tensor(x_mask).to(self.device)

        num_atom_features = x_atom.shape[-1]
        num_bond_features = x_bonds.shape[-1]
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = (
            get_smiles_array(data_gat.smiles.values, data_gat_feature_dicts)
        )
        x_atom = tc.Tensor(x_atom).to(self.device)
        x_bonds = tc.Tensor(x_bonds).to(self.device)
        x_atom_index = tc.cuda.LongTensor(x_atom_index).to(self.device)
        x_bond_index = tc.cuda.LongTensor(x_bond_index).to(self.device)
        x_mask = tc.Tensor(x_mask).to(self.device)

        self.gat_encoder = GAT(
            self.radius,
            self.T,
            num_atom_features,
            num_bond_features,
            self.fingerprint_dim,
            self.p_dropout,
        ).to(self.device)

        graph_embedding = self.gat_encoder(
            x_atom,
            x_bonds,
            x_atom_index,
            x_bond_index,
            x_mask,
        )
        graph_embedding.to(self.device)

        fingerprint_embedding = self.fingerprint_encoder.forward(
            data_batch1["input"]["smiles"]
        )
        fingerprint_embedding.to(self.device)

        molecule_embedding1 = molecule_embedding[:, 0, :]
        self.log_embedding_shapes(
            molecule_embedding1, graph_embedding, fingerprint_embedding
        )

        x1 = tc.cat(
            [molecule_embedding1, graph_embedding, fingerprint_embedding],
            dim=-1,
        )
        x2 = self.mlp(x1)
        os.remove(csv_file_path)
        return x2


def evaluate(model, data_loader, csv_save):
    model.eval()
    label_predict = tc.tensor([], dtype=tc.float32).cuda()
    label_true = tc.tensor([], dtype=tc.long).cuda()
    with tc.no_grad():
        for data_batch in data_loader:
            label_predict_batch = model(data_batch)

            label_true_batch = data_batch['target']['label'].to(tc.long)
            label_predict = tc.cat((label_predict, label_predict_batch.detach()), dim=0)
            label_true = tc.cat((label_true, label_true_batch.detach()), dim=0)

    label_predict = tc.softmax(label_predict, dim=1)
    label_predict = label_predict.cpu().numpy()
    predict_label = np.argmax(label_predict, axis=1)
    label_true = label_true.cpu().numpy()

    
    if csv_save == True:
        df = pd.DataFrame({'label_true': label_true, 'predict_label': predict_label, 'label_predict': label_predict[:, 1]})
        df.to_csv('label_predict_test_0.csv', index=False)

    auc_roc = round(roc_auc_score(label_true, label_predict[:, 1]), 3)
    auc_prc = round(average_precision_score(label_true, label_predict[:, 1]), 3)
    accuracy = round(accuracy_score(label_true, np.argmax(label_predict, axis=1)), 3)
    precision = round(precision_score(label_true, np.argmax(label_predict, axis=1)), 3)
    recall = round(recall_score(label_true, np.argmax(label_predict, axis=1)), 3)
    f1_score = round(2 * precision * recall / (precision + recall), 3)
    jaccard = round(jaccard_score(label_true, np.argmax(label_predict, axis=1)), 3)
    mcc = round(matthews_corrcoef(label_true, np.argmax(label_predict, axis=1)), 3)
    balanced_accuracy = round(balanced_accuracy_score(label_true, np.argmax(label_predict, axis=1)), 3)

    metric = {
        "auc_roc": auc_roc,
        "auc_prc": auc_prc,
        "mcc": mcc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "jaccard": jaccard,
    }

    return metric


def train(trial_version, epochs):
    data_loader_train, data_loader_validate, data_loader_test = convert_data_list_to_data_loader()
    model = SynthMolClassifier()
    model.to(device)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15)

    current_best_metric = -1e10
    no_change_count = 0

    last_loss = None  
    tolerance = 1e-6  

    for epoch in range(epochs):
        model.train()
        for step, data_batch in enumerate(data_loader_train):
            label_predict_batch = model(
                data_batch,   
            )
            label_true_batch = data_batch['target']['label'].to(tc.long)

            loss = criterion(label_predict_batch, label_true_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 19 == 0:
                print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, round(loss.item(), 3)))

            if last_loss is not None and abs(loss.item() - last_loss) < tolerance:
                no_change_count += 1
            else:
                no_change_count = 0
            last_loss = loss.item()

            if no_change_count > 10:
                print("Early stopping!")
                break

        scheduler.step()

        metric_train = evaluate(model, data_loader_train, csv_save=False)
        metric_validate = evaluate(model, data_loader_validate, csv_save=False)
        metric_test = evaluate(model, data_loader_test, csv_save=True)
        logging.info(
            f"Epoch: {epoch}, Train metrics: {metric_train}, Validate metrics: {metric_validate}, Test metrics: {metric_test}"
        )
        tc.save(model.state_dict(), f"./weight/{trial_version}_{epoch}.pt")
        if metric_validate["auc_roc"] > current_best_metric:
            current_best_metric = metric_validate["auc_roc"]
            current_best_epoch = epoch
            tc.save(model.state_dict(), f"./weight/{trial_version}.pt")


if __name__ == "__main__":
    set_random_seed(1024)
    print("data_process start!") 
    calculate_molecule_3D_structure()
    construct_data_list()

    print("train start!") 
    
    train(trial_version='1',epochs=100) 

    print('All is well!')
