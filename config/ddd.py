parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=70)  # 500200 batches at bs 16, 117263 images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=2)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-visdrone.cfg', help='cfg file path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--transfer', action='store_true', help='transfer learning')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='/home/twsf/.cache/torch/checkpoints/darknet53.conv.74', help='initial weights')  # i.e. weights/darknet.53.conv.74
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # defaultpw, uCE, uBCE
    parser.add_argument('--prebias', action='store_true', help='transfer-learn yolo biases prior to training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--var', type=float, help='debug variable')
    parser.add_argument('--root_path', type=str, default="/home/twsf/data/Visdrone/detect_voc", help='path of datasets')
    parser.add_argument('--visdom', default=False, type=bool, help='Use visdom for loss visualization')
    opt = parser.parse_args()
    
    
    

# classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
#            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
#            'tvmonitor')

classes = ('pedestrian', 'people', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')



    else:  # Evolve hyperparameters (optional)
        opt.notest = True  # only test final epoch
        opt.nosave = True  # only save final checkpoint
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                x = np.loadtxt('evolve.txt', ndmin=2)
                if len(x) > 1:
                    parent = 'weighted'  # parent selection method: 'single' or 'weighted'
                    if parent == 'single':
                        x = x[fitness(x).argmax()]
                    elif parent == 'weighted':  # weighted combination
                        n = min(10, x.shape[0])  # number to merge
                        x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                        w = fitness(x) - fitness(x).min()  # weights
                        x = (x[:n] * w.reshape(n, 1)).sum(0) / w.sum()  # new parent
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = x[i + 7]

                # Mutate
                np.random.seed(int(time.time()))
                s = [.1, .1, .1, .1, .1, .1, .1, .0, .02, .2, .2, .2, .2, .2, .2, .2, .2]  # sigmas
                for i, k in enumerate(hyp.keys()):
                    x = (np.random.randn(1) * s[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
                    hyp[k] *= float(x)  # vary by sigmas

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train()

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)
