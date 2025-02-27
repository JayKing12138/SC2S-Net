class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool3d(1)#NxCx1x1x1
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.max_a = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // 9, 1, bias=False),
            nn.PReLU(),
            nn.Conv3d(in_planes // 9, in_planes, 1, bias=False)
        )

        self.avg_a = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // 9, 1, bias=False),
            nn.PReLU(),
            nn.Conv3d(in_planes // 9, in_planes, 1, bias=False)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.max_pool(x)
        x2 = self.avg_pool(x)
        max_out = self.max_a(x1)
        avg_out = self.avg_a(x2)
        ma_out = max_out + avg_out
        out = self.sig(ma_out)
        out = x.mul(out)
        return out

class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        # Voxel-level and channel-level fusion
        self.Attentionxq = nn.Sequential(
            ChannelAttention(90),
            nn.Conv3d(90, 45, kernel_size=1, bias=True),
            nn.PReLU()
        )
        self.Attentionx = nn.Sequential(
            ChannelAttention(45),
            nn.Conv3d(45, 45, kernel_size=1, bias=True),
            nn.PReLU()
        )
        self.Attentionq = nn.Sequential(
            ChannelAttention(45),
            nn.Conv3d(45, 45, kernel_size=1, bias=True),
            nn.PReLU()
        )
        self.meta_learner1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.meta_learner2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.meta_learner3 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(45, 45, kernel_size=1, bias=False)
        )

    def forward(self, pred1, pred2):
        # Concatenate the predictions
        xq = torch.cat((pred1, pred2), dim=1)
        xq_features = self.Attentionxq(xq)

        x_features = self.Attentionx(pred1)
        q_features = self.Attentionq(pred2)
        weights1 = self.meta_learner1(xq_features)
        weights2 = self.meta_learner2(x_features)
        weights3 = self.meta_learner3(q_features)
        xq_fuse = weights1 * xq_features + weights2 * x_features + weights3 * q_features

        #xq_f