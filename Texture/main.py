from __future__ import print_function, division

from model import *
from utils import *
from dataset import *
# from torchvision.utils import make_grid
from sklearn.manifold import TSNE
import torch.optim as optim

plt.switch_backend('agg')


def FeatureMap2Heatmap(inputs, feature, map_x, spoof_label):
    plt.figure()
    feature.insert(0, inputs)
    id = 0 if spoof_label[0] == 1 else abs(spoof_label[0])
    label = ['real', 'photo', 'video'][id]
    for n, feat in enumerate(feature):
        feature_first_frame = feat[0, :, 0, :, :].cpu() if len(feat.shape) == 5 else feat[0, :, :, :].cpu()
        heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
        for i in range(feature_first_frame.size(0)):
            heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                       feature_first_frame.size(2))
        heatmap = heatmap.data.numpy()
        plt.subplot(3, 3, n + 1)
        plt.imshow(heatmap)
        plt.colorbar()
        plt.title(label if n == 0 else str(n - 1))

    heatmap2 = torch.pow(map_x[0, 0, :, :] if len(map_x.shape) == 4 else map_x[0, :, :], 2)
    heatmap2 = heatmap2.data.cpu().numpy()

    plt.subplot(3, 3, n + 2)
    plt.imshow(heatmap2)
    plt.colorbar()
    plt.title('DepthMap')
    plt.tight_layout()
    plt.savefig(args.log + '/feature_visual%d%s.jpg' % (args.resume, label))
    plt.close()


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)

    input = input.expand(input.shape[0], 8, input.shape[2], input.shape[3])

    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv

    return contrast_depth


def showloss(epochs, loss1, loss2, epochs_val, val_acers, test_acers, lrs):
    plt.figure(1)
    plt.suptitle('Loss and ACER')
    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss1, color='r', linewidth=2, label='absolute')
    plt.plot(epochs, loss2, color='pink', linewidth=2, label='contrast')
    plt.ylabel('loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(epochs[-min(75, len(epochs)):], loss1[-min(75, len(epochs)):], color='r', linewidth=2,
             label='absolute')
    plt.plot(epochs[-min(75, len(epochs)):], loss2[-min(75, len(epochs)):], color='pink', linewidth=2,
             label='contrast')
    plt.subplot(2, 2, 3)
    plt.plot(epochs_val, val_acers, color='b', marker='.', linewidth=1, label='val')
    plt.plot(epochs_val, test_acers, color='g', marker='.', linewidth=1, label='test')
    plt.xlabel('epochs')
    plt.ylabel('ACER')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(epochs_val[-min(5, len(epochs_val)):], val_acers[-min(5, len(epochs_val)):], color='b', marker='.',
             linewidth=1, label='val')
    plt.plot(epochs_val[-min(5, len(epochs_val)):], test_acers[-min(5, len(epochs_val)):], color='g', marker='.',
             linewidth=1, label='test')
    plt.xlabel('epochs')
    plt.savefig('./%s/%s.pdf' % (args.log, args.log))
    plt.close()


class Contrast_depth_loss(nn.Module):
    def __init__(self):
        super(Contrast_depth_loss, self).__init__()
        return

    def forward(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)

        criterion_MSE = nn.MSELoss()
        loss = criterion_MSE(contrast_out, contrast_label)
        return loss


# main function
def train_test():
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    init_logging(log_file=args.log + '/log.txt', file_mode='a', overwrite_flag=True)

    model = T_Net()
    model = nn.DataParallel(model).to(device)

    # resume or scratch
    epochs, loss1, loss2, epochs_val, val_acers, test_acers, lrs = [], [], [], [], [], [], []
    if args.resume:
        logging.info('\nResume %d!' % args.resume)
        resume_data = torch.load('./%s/#epoch%s.pth.tar' % (args.log, args.resume))
        model.load_state_dict(resume_data['state_dict'])
        epochs, loss1, loss2, epochs_val, val_acers, test_acers, lrs = resume_data['results']
        showloss(epochs, loss1, loss2, epochs_val, val_acers, test_acers, lrs)
    else:
        logging.info('\nTrain from scratch!')
        logging.info(model)
    logging.info('train_image: %s' % train_image_dir)
    logging.info('train_protocol: %s' % train_list)
    for key in args.__dict__.keys():
        logging.info('%s: %s' % (key, getattr(args, key)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    criterion_absolute_loss = nn.MSELoss()
    criterion_contrastive_loss = Contrast_depth_loss().to(device)

    # func: val and test
    def val_test():
        model.eval()
        t = time.time()
        with torch.no_grad():
            '''                  val for threshold                 '''
            print('valing')
            map_score_dict = {}
            for start in range(3):
                val_data = Spoofing_train(val_list, val_image_dir, val_map_dir, args,
                                          transform=transforms.Compose([ToTensor(), Normaliztion()]))
                dataloader_val = DataLoader(val_data, batch_size=args.batchsize, shuffle=False, num_workers=4)
                res = torch.tensor([])
                label = []
                video = []
                for i, sample_batched in enumerate(dataloader_val):
                    inputs, map_label, spoof_label, video_names = sample_batched
                    inputs = inputs.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    map_x, feature = model(inputs)

                    map_score = map_x.mean(tuple(range(1, len(map_x.shape))))  # Calculate the mean value of each map
                    res = torch.cat((res, map_x.view(map_x.shape[0], -1).cpu()), 0)
                    label.extend(spoof_label.tolist())
                    video.extend(video_names)
                    for n, path in enumerate(video_names):
                        if path in map_score_dict:
                            map_score_dict[path].append([map_score[n], spoof_label[n]])
                        else:
                            map_score_dict[path] = [[map_score[n], spoof_label[n]]]
            map_score_list = []
            val_score = []
            for k in map_score_dict:
                value = np.array(map_score_dict[k]).mean(0)
                map_score_list.append('{} {} {}\n'.format(k, value[0], value[1]))
                val_score.append(value[0].item())
            val_score.sort()
            cutoff = [val_score[len(val_score) // 10]]
            map_score_val_filename = args.log + '/' + args.log + '_map_score_val.txt'
            with open(map_score_val_filename, 'w') as file:
                file.writelines(map_score_list)

            '''                 test for performance              '''
            print('testing')
            map_score_dict = {}
            for start in range(5):
                test_data = Spoofing_train(test_list, test_image_dir, test_map_dir, args,
                                           transform=transforms.Compose([ToTensor(), Normaliztion()]))
                dataloader_test = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=4)
                res_all = res.clone()
                label_all = label.copy()
                video_all = video.copy()
                for i, sample_batched in enumerate(dataloader_test):
                    inputs, map_label, spoof_label, video_names = sample_batched
                    inputs = inputs.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    map_x, feature = model(inputs)

                    map_score = map_x.mean(tuple(range(1, len(map_x.shape))))
                    res_all = torch.cat((res_all, map_x.view(map_x.shape[0], -1).cpu()), 0)
                    label_all.extend([x + 4 for x in spoof_label])  # +4 for distinguishing the label of val
                    video_all.extend(video_names)
                    for n, path in enumerate(video_names):
                        if path in map_score_dict:
                            map_score_dict[path].append([map_score[n], spoof_label[n]])
                        else:
                            map_score_dict[path] = [[map_score[n], spoof_label[n]]]
            map_score_list = []
            for k in map_score_dict:
                value = np.array(map_score_dict[k]).mean(0)
                map_score_list.append('{} {} {}\n'.format(k, value[0], value[1]))
            map_score_test_filename = args.log + '/' + args.log + '_map_score_test.txt'
            with open(map_score_test_filename, 'w') as file:
                file.writelines(map_score_list)

            '''       tsne for both val and test           '''
            tsne = TSNE(n_components=2)
            Y = tsne.fit_transform(res_all)
            np.savez('%s/test_tsne.npz' % args.log, Y=Y, label_all=np.array(label_all), name=video_all)
            plot_embedding(Y, label_all, '%s/tsne_all_%s.pdf' % (args.log, epoch + 1), name=video_all)

            '''       performance measurement both val and test           '''
            val_APCER, val_BPCER, val_ACER, val_threshold, test_APCER, test_BPCER, test_ACER, test_threshold_ACER = performances(
                map_score_val_filename, map_score_test_filename)

            logging.info('[epoch:%d] Time: %.3fs Val: threshold= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (
                epoch + 1, time.time() - t, val_threshold, val_APCER, val_BPCER, val_ACER))
            logging.info('[epoch:%d] Test_Threshold_ACER= %.4f   Test: APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (
                epoch + 1, test_threshold_ACER, test_APCER, test_BPCER, test_ACER))
        return val_ACER, test_ACER, cutoff

    # test
    if args.test:
        logging.info('Test')
        epoch = args.resume - 1
        val_test()
        return

    # train
    train_data = Spoofing_train(train_list, train_image_dir, train_map_dir, args, transform=transforms.Compose(
        [RandomHorizontalFlip(), ToTensor(), Normaliztion()]))
    dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    for epoch in range(args.epochs):
        scheduler.step()
        if epoch < args.resume:
            continue
        lrs.append(optimizer.param_groups[0]['lr'])
        loss_absolute = AvgrageMeter()
        loss_contra = AvgrageMeter()

        model.train()
        t0 = t1 = time.time()
        for i, sample_batched in enumerate(dataloader_train):
            inputs, map_label, spoof_label, path = sample_batched
            inputs = inputs.to(device, non_blocking=True)
            map_label = map_label.unsqueeze(dim=1).to(device, non_blocking=True)

            optimizer.zero_grad()
            map_x, feature = model(inputs)

            contrastive_loss = criterion_contrastive_loss(map_x, map_label)
            absolute_loss = criterion_absolute_loss(map_x, map_label)
            loss = absolute_loss + contrastive_loss
            loss.backward()
            optimizer.step()

            loss_absolute.update(absolute_loss.data, inputs.size(0))
            loss_contra.update(contrastive_loss.data, inputs.size(0))

            if i % args.echo_batches == args.echo_batches - 1:
                FeatureMap2Heatmap(inputs, feature, map_x, spoof_label)
                print(
                    '[%d] batch:%3d, lr=%.3e, Absolute_loss= %f, Contrastive_loss= %f batch_time: %.3fs' % (
                        epoch + 1, i + 1, lrs[-1], loss_absolute.avg, loss_contra.avg,
                        time.time() - t0))
                t0 = time.time()
        print(
            'Epoch:%d, lr=%.3e, Train: Absolute_loss= %f, Contrastive_loss= %f epoch_time: %.3f\n' % (
                epoch + 1, lrs[-1], loss_absolute.avg, loss_contra.avg, time.time() - t1))
        loss1.append(loss_absolute.avg)
        loss2.append(loss_contra.avg)
        loss_absolute.reset()
        loss_contra.reset()
        epochs.append(epoch + 1)
        # validation
        if epoch % args.epoch_test == args.epoch_test - 1:
            logging.info('[epoch:%d] Val every %d training epochs, lr=%.3e' % (epoch + 1, args.epoch_test, lrs[-1]))
            val_ACER, test_ACER, cutoff = val_test()
            val_acers.append(val_ACER)
            test_acers.append(test_ACER)
            epochs_val.append(epoch + 1)
            torch.save({
                'results': [epochs, loss1, loss2, epochs_val, val_acers, test_acers, lrs],
                'cutoff': cutoff,
                'state_dict': model.state_dict(),
            }, './%s/#epoch%d.pth.tar' % (args.log, epoch + 1))
            print('success: save model %d' % (epoch + 1))
        showloss(epochs, loss1, loss2, epochs_val, val_acers, test_acers, lrs)
    print('Finished Training')


# Dataset root
data_root = '/home/yaowen/Documents/1Database/PAD-datasets/'

train_image_dir = data_root + 'Oulu-NPU/Aligned_faces_x1.6/Train'
val_image_dir = data_root + 'Oulu-NPU/Aligned_faces_x1.6/Dev'
test_image_dir = data_root + 'Oulu-NPU/Aligned_faces_x1.6/Test'

train_map_dir = data_root + 'Oulu-NPU/Aligned_depthmap_x1.6/Train'
val_map_dir = data_root + 'Oulu-NPU/Aligned_depthmap_x1.6/Dev'
test_map_dir = data_root + 'Oulu-NPU/Aligned_depthmap_x1.6/Test'

train_list = './Protocol/Protocol_4/Train_2.txt'
val_list = './Protocol/Protocol_4/Dev_2.txt'
test_list = './Protocol/Protocol_4/Test_2.txt'

# train_list = './Protocol/Protocol_1/Train.txt'
# val_list = './Protocol/Protocol_1/Dev.txt'
# test_list = './Protocol/Protocol_1/Test.txt'

if __name__ == "__main__":
    class Argparse(object):
        def __init__(self):
            self.gpu = '0'
            self.lr = 0.001
            self.batchsize = 16
            self.milestones = [100, 200, 300]
            self.gamma = 0.1
            self.echo_batches = 10
            self.epoch_test = 10
            self.epochs = 400
            self.mapsize = 32
            self.inputsize = 256
            self.log = 'T_P4-2'
            self.resume = 260
            self.test = True


    args = Argparse()
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    train_test()
