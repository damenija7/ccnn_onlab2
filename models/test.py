import torch

class Test0(torch.nn.Module):
    def __init__(self, in_channel=1024):
        super().__init__()

        self.head = torch.nn.Linear(in_features=in_channel, out_features=1, bias=False)
    def forward(self, x):
        return torch.nn.functional.sigmoid(self.head(x).squeeze(dim=-1))


class Test3(torch.nn.Module):
    def __init__(self, in_channel=1024, out_channel=1):
        super().__init__()

        self.head1 = torch.nn.Sequential(torch.nn.Linear(in_features=in_channel, out_features=in_channel//2, bias=True),
                                        torch.nn.ReLU())
        self.batch1 = torch.nn.BatchNorm1d(num_features=in_channel//2)


        in_channel = in_channel//2

        self.head2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channel, out_features=in_channel, bias=True),
            torch.nn.ReLU())
        self.batch2 = torch.nn.BatchNorm1d(num_features=in_channel)

        self.head3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channel, out_features=in_channel // 2, bias=True),
            torch.nn.ReLU())
        self.batch3 = torch.nn.BatchNorm1d(num_features=in_channel // 2)

        self.tail = torch.nn.Linear(in_features=in_channel // 2, out_features=1, bias=True)

    def forward(self, x):
        x = torch.permute(self.batch1(torch.permute(self.head1(x), (0,2,1))), (0,2,1))

        x = x + torch.permute(self.batch2(torch.permute(self.head2(x), (0,2,1))), (0,2,1))

        x = torch.permute(self.batch3(torch.permute(self.head3(x), (0,2,1))), (0,2,1))

        x = self.tail(x)

        return torch.nn.functional.sigmoid(x)


class Test2(torch.nn.Module):
    def __init__(self, in_channel=1024):
        super().__init__()

        self.head = torch.nn.Sequential(torch.nn.Linear(in_features=in_channel, out_features=in_channel//2, bias=True),
                                        torch.nn.ReLU())
        self.batch = torch.nn.BatchNorm1d(num_features=in_channel//2)
        self.tail =  torch.nn.Linear(in_features=in_channel//2, out_features=1, bias=True)

    def forward(self, x):
        x = self.head(x)
        x = torch.permute(x, (0,2,1))
        x = self.batch(x)
        x = torch.permute(x, (0,2,1))
        x = self.tail(x)

        return torch.nn.functional.sigmoid(x.squeeze(dim=-1))


class TestConv2(torch.nn.Module):
    def __init__(self, in_channels = 1024):
        super().__init__()

        self.conv = torch.nn.Sequential(torch.nn.Conv1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, padding=3//2),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(num_features=in_channels//2),

                                    torch.nn.Conv1d(in_channels=in_channels//2, out_channels=in_channels//4, kernel_size=3, padding=3//2),
                                    torch.nn.ReLU(),
                                    torch.nn.BatchNorm1d(num_features=in_channels//4),


                                    torch.nn.Conv1d(in_channels=in_channels//4, out_channels=1, kernel_size=3, padding=3//2))
        self.tail = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.permute(x, (0,2,1))
        x = self.conv(x)
        x = torch.permute(x, (0, 2,1))
        x = self.tail(x)
        return torch.squeeze(x, dim=-1)


class TestConv(torch.nn.Module):
    def __init__(self, in_channels = 1024):
        super().__init__()

        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=3//2)
        self.tail = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.permute(x, (0,2,1))
        x = self.conv(x)
        x = self.tail(x)
        x = torch.permute(x, (0, 2,1))
        return torch.squeeze(x, dim=-1)




class TestConv3(torch.nn.Module):
    def __init__(self, in_channels = 1024, *args):
        super().__init__()

        kern_size = 7

        self.conv = torch.nn.Sequential(torch.nn.Conv1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=kern_size, padding=kern_size//2, padding_mode='circular'),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(num_features=in_channels//2),
                                    torch.nn.Conv1d(in_channels=in_channels//2, out_channels=in_channels//4, kernel_size=kern_size, padding=kern_size//2, padding_mode='circular'),
                                    torch.nn.ReLU(),
                                    torch.nn.BatchNorm1d(num_features=in_channels//4),
                                    torch.nn.Conv1d(in_channels=in_channels//4, out_channels=1, kernel_size=kern_size, padding=kern_size//2, padding_mode='circular'))



        self.tail = torch.nn.Softmax(dim=-1)

    def forward(self, x, label=None):
        x = torch.permute(x, (0,2,1))
        x = self.conv(x)
        x = torch.permute(x, (0, 2,1))
        x = self.tail(x)
        return x