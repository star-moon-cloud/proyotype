import time
import torch

from helpers import list_of_distances, make_one_hot
from settings import prototype_shape as ps
from settings import inst_shape,num_classes
def _train_or_test(model, dataloader,dataloader1=None,optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    n_batches1 = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_cluster_cost1 = 0
    total_inst_cost = 0

    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    '''
    for i, (image, label) in enumerate(dataloader1):
        input = image.cuda()
        target = label.cuda()
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            #output, min_distances,sim = model(input)
            min_distances = model.module.inst_forward(input)
            #min_distances.retain_grad()
            #print("min_distances = ",min_distances)
            # compute loss
            if class_specific:
                max_dist = (model.module.prototype_shape[1]  #128
                            * model.module.prototype_shape[2]  #1
                            * model.module.prototype_shape[3])  #1
                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate inst cost

                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity2[:,label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                inst_cost =  torch.mean(max_dist - inverted_distances)
                #print("inst_cost.shape",inst_cost.shape,"\ninst_cost",inst_cost)
                #print("inverted_distances  =  ",inverted_distances.shape)
                #print("_  =  ", _)
            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                inst_cost = torch.mean(min_distance)
            n_batches1 = n_batches1+1
            total_inst_cost = total_inst_cost+inst_cost.item()
        del input
        del target
        del min_distances
        '''
    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            '''
            min_distances_clst,min_distances_inst = model.module.clst_forward(input)
            #print("min_distances = ",min_distances)
            # compute loss
            

            if class_specific:
                max_dist = (model.module.prototype_shape[1]  #128
                            * model.module.prototype_shape[2]  #1
                            * model.module.prototype_shape[3])  #1
                #print("max_distances  =  ",max_dist)
                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                #print("label = ", label)
                #print("model.module.prototype_class_identity",model.module.prototype_class_identity.shape)
                #print("model.module.prototype_class_identity[:,label]",model.module.prototype_class_identity[:,label].shape)
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity1[:,label]).cuda()
                #print("prototypes_of_correct_class  =  ",prototypes_of_correct_class.shape)# 80*100
                #print(torch.max((max_dist - min_distances) * prototypes_of_correct_class).shape,"----------------",torch.max(max_dist - min_distances))
                inverted_distances, _ = torch.max((max_dist - min_distances_clst) * prototypes_of_correct_class, dim=1)#每行最大值和行索引
                
                #print("inverted_distances  =  ",inverted_distances.shape)
                #print("_  =  ", _)
                cluster_cost = torch.mean(max_dist - inverted_distances)
                #print("cluster_cost  =  ",cluster_cost)
                prototypes_of_correct_class1 = torch.t(model.module.prototype_class_identity2[:,label]).cuda()
                cluster_cost1 = torch.mean(min_distances_inst* prototypes_of_correct_class1)
                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances_clst) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
                #print("separation_cost",separation_cost)
                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances_clst * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                #print("avg_separation_cost",avg_separation_cost)
                '''
            if class_specific:
                max_dist = (model.module.prototype_shape[1]  #128
                            * model.module.prototype_shape[2]  #1
                            * model.module.prototype_shape[3])  #1
                #print("max_distances  =  ",max_dist)
                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                #print("label = ", label)
                #print("model.module.prototype_class_identity",model.module.prototype_class_identity.shape)
                #print("model.module.prototype_class_identity[:,label]",model.module.prototype_class_identity[:,label].shape)
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                #print("prototypes_of_correct_class  =  ",prototypes_of_correct_class.shape)# 80*100
                #print(torch.max((max_dist - min_distances) * prototypes_of_correct_class).shape,"----------------",torch.max(max_dist - min_distances))
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)#每行最大值和行索引
                
                #print("inverted_distances  =  ",inverted_distances.shape)
                #print("_  =  ", _)
                cluster_cost = torch.mean(max_dist - inverted_distances)
                #print("cluster_cost  =  ",cluster_cost)
                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
                #print("separation_cost",separation_cost)
                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    #print("l1 mask  = ",l1_mask.shape)
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                    #print("l1  = ",l1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                print("else>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                min_distance, _ = torch.min(min_distances, dim=1)
                #min_distance1, _ = torch.min(min_distances_inst, dim=1)
                cluster_cost = torch.mean(min_distance)
                #cluster_cost1 = torch.mean(min_distance1)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples = n_examples+target.size(0)
            n_correct = n_correct+(predicted == target).sum().item()
            #print("sim",sim.item())
            n_batches = n_batches+1
            total_cross_entropy = total_cross_entropy+cross_entropy.item()
            total_cluster_cost = total_cluster_cost+cluster_cost.item()
            #total_cluster_cost1 = total_cluster_cost1+cluster_cost1.item()
            total_separation_cost = total_separation_cost+separation_cost.item()
            total_avg_separation_cost = total_avg_separation_cost+avg_separation_cost.item()
            #total_sim=total_sim+sim.item()
        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          #+ coefs['clst1'] * cluster_cost1
                          + coefs['sep'] * separation_cost
                          #+ coefs['inst'] * total_inst_cost/ n_batches1
                          + coefs['l1'] * l1
                          #+ coefs['sim'] * sim.item()
                          )
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1 #+ 1 * total_inst_cost/ n_batches1
                    #print("LOSS = ",loss)
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          #+ coefs['clst1'] * cluster_cost1
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            #print("LOSS = ",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances
        #del min_distances_clst
        #del min_distances_inst

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    #log('\tcluster1: \t{0}'.format(total_cluster_cost1 / n_batches))
    #log('\tinst: \t{0}'.format(total_inst_cost / n_batches1))
    #log('\tsim: \t{0}'.format(sim.item()))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(model, dataloader,dataloader1, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader,dataloader1=dataloader1, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader,dataloader1, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader,dataloader1=dataloader1, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
