import torch
import torch.distributed as dist
import argparse

import torchvision.transforms as transforms

from torch.utils.tensorbord import SummaryWriter

parser=argparse.ArgumentParser(description='')
parser.add_argument('--start',type=int,default=1,help='')
parser.add_argument('--epoch',type=int,default=100,help='')

parser.add_argument('--accum',type=int,default=10,help='')

parser.add_argument('--max_norm',type=float,default=5.0,help='')

parser.add_argument('--distributed',action='store_false',help='')
parser.set_defaults(distributed=True)

parser.add_argument('--pin_mem',action='store_false',help='')
parser.set_defaults(pin_mem=True)

parser.add_argument('--batch_size',type=int,default=64,help='')

parser.add_argument('--device',default='cuda',help='')

parser.add_argument('--lr',type=float,default=0.2,help='')

parser.add_argument('--blr',type=float,default=0.003,help='')

parser.add_argument('--min_lr',type=float,default=1e-6,help='')

parser.add_argument('--warmup_epochs',type=int,default=5,help='')

parser.add_argument('--log_dir',default='/tmp/train/logs',help='')
parser.add_argument('--data',default='/tmp/train/data',help='')

args=parser.parser_args()

logs=SummaryWriter(log_dir=args.log_dir)
devices=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

## adjust learning rate，linear if epoch <warmup epoch.cos if epoch>=warmup_epoch
def adjust_lr(optimizer,epoch,args):
    if epoch < args.warmup_epochs:
        lr=args.lr*epoch/args.warmup_epochs
    else:
        lr=args.min_lr+(args.lr-args.min_lr)*0.5*(1.+math.cos(math.pi*(epoch-args.warmup_epochs)/(args.epochs-args.warmup_epochs)))
    for param in otpimizer.param_groups:
        if "lr_scale" in param:
            param['lr']=lr*param['lr_scale']
        else:
            param['lr']=lr
    return lr

model=Model()
model.to(devices)

transforms=transforms.Compose([
    transforms.RandomResizedCrop(args.input_size,scale=(0.08,1.0),interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.toTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
dataset=Dataset(root=args.data,transform=transforms)

if args.distributed:
    model=torch.nn.parellel.DistributedDataParellel(model,device_ids=[args.gpu],find_unused_parameters=True)
    word_size=dist.get_world_size()
    global_rank=dist.get_rank()
    args.lr=args.blr*word_size*args.batch_size/256.
    sampler=torch.utils.data.DistributedSampler(dataset,num_replicas=word_size,rank=global_rank,shuffle=True)
else:
    sampler=torch.utils.data.RandomSampler(dataset)
    
criterion=torch.nn.CrossEntropyLoss()

dataloader=torch.utils.data.DataLoader(dataset,sampler=sampler,
                                    pin_memory=args.pin_mem,
                                    num_workers=10,
                                    batch_size=args.batch_size,
                                    drop_last=True)


optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.999))

for epoch in range(args.start,args.epoch):
    model.train(True)
    if args.distributed:
        dataloader.sampler.set_epoch(epoch)
    for idx,data,target in enumerate(dataloader):
        #adjust learning rate
        if (idx+1)%args.accum==0 or (idx+1)==len(dataloader):
            adjust_lr(optimizer,(idx+1)/len(dataloader)+epoch,args)

        data=data.to(devices,non_blocking=True)
        target=target.to(devices,non_blocking=True)

        out=model(data)
        loss=criterion(out,target)

        loss_value=loss.item()

        loss.backward()

        ##clip grad by torch.nn.utils.clip_grad_norm()
        torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=args.max_norm)

        ##get max learning rate to write summary
        min_lr=0.0
        max_lr=10.0
        for group in optimizer.param_groups:
            min_lr=min(minlr,group['lr'])
            max_lr=max(maxlr,group['lr'])
            
        loss=loss/args.accum
        
        if idx%args.accum==0 or idx+1==len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            
            if word_size>1:
                reduce_all_loss=torch.tensor(loss_value).cuda()
                dist.all_reduce(reduce_all_loss)
                reduce_all_loss/=word_size

            iters=idx/len(dataloader)+epoch
            iters=int(iters*1000)
            logs.add_scalar('loss',reduce_all_loss.item(),iters)
            logs.add_scalar('lr',max_lr,iters)

    torch.save({'net':model.parameters(),'optimizer':optimizer.state_dict(),'args':args},os.path.join(args.path,'epoch-{}.pth'.format(args.epoch)))
