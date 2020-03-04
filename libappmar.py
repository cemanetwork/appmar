"""
APPMAR 1.0
A toolbox for management of meteorological and marine data on limited information regions

Marianella Bolívar
Diego Casas
Germán Rivillas Ospina, PhD
"""

import os
import wx
import matplotlib
from libappmar import download_data, frequency_curve, joint_distribution, load_data, load_obj, save_obj, merge_data, weibull_data, load_max, interp_idw
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as NavigationToolbar
from matplotlib.figure import Figure
from scipy import stats
import numpy as np
from windrose import WindroseAxes
import matplotlib.pyplot as plt
import weibull
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xarray as xr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

LAND_10M = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor="k", facecolor="grey")
N = 181

matplotlib.rcParams["font.family"] = "Times New Roman"

MONTHS = {
    "Winter": [12, 1, 2],
    "Summer": [6, 7, 8],
    "Spring": [3, 4, 5],
    "Fall": [9, 10, 11],
    "All": [*range(1, 13)]
}

STR_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

class FrameCanvas(wx.Frame):
    def __init__(self, *args, **kw):
        super(FrameCanvas, self).__init__(*args, **kw)

        self.fig = Figure()
        self.canvas = FigureCanvas(self, -1, self.fig)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        self.SetSizer(self.sizer)
        self.Fit()

        self.add_toolbar()  # comment this out for no toolbar

    def add_toolbar(self):
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()
        # By adding toolbar in sizer, we are able to put it at the bottom
        # of the frame - so appearance is closer to GTK version.
        self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        # update the axes menu on the toolbar
        self.toolbar.update()


class FrameDownloadWaves(wx.Frame):
    """
    TO DO
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameDownloadWaves, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Welcome to Wave Download Module")
        txt_grids = wx.StaticText(pnl, label="Grid IDs (comma-separated):")
        txt_years = wx.StaticText(pnl, label="Year range (comma-separated):")

        # put a text entry for grid IDs
        self.ent_grids = wx.TextCtrl(pnl, value="wna,at_10m")
        self.ent_years = wx.TextCtrl(pnl, value="1999,2018")

        # put some buttons
        btn_winter = wx.Button(pnl, label="Winter")
        btn_summer = wx.Button(pnl, label="Summer")
        btn_spring = wx.Button(pnl, label="Spring")
        btn_fall = wx.Button(pnl, label="Fall")
        btn_whole = wx.Button(pnl, label="Whole year")
        btn_exit = wx.Button(pnl, label="Exit")

        # associate a handler function to the buttons
        btn_winter.Bind(wx.EVT_BUTTON, self.on_winter)
        btn_summer.Bind(wx.EVT_BUTTON, self.on_summer)
        btn_spring.Bind(wx.EVT_BUTTON, self.on_spring)
        btn_fall.Bind(wx.EVT_BUTTON, self.on_fall)
        btn_whole.Bind(wx.EVT_BUTTON, self.on_whole)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        ### TO DO ###
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(txt_grids, sizer_flags)
        sizer.Add(self.ent_grids, sizer_flags)
        sizer.Add(txt_years, sizer_flags)
        sizer.Add(self.ent_years, sizer_flags)
        sizer.Add(btn_winter, sizer_flags)
        sizer.Add(btn_summer, sizer_flags)
        sizer.Add(btn_spring, sizer_flags)
        sizer.Add(btn_fall, sizer_flags)
        sizer.Add(btn_whole, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def download_data(self, months):
        """Download data for the given months."""
        gids = self.ent_grids.GetLineText(0).split(",")
        pids = ["hs", "tp", "dp"]
        yi, yf = self.ent_years.GetLineText(0).split(",")
        ys = [*range(int(yi), int(yf) + 1)]
        progress_dlg = wx.ProgressDialog(
            "Download", "Downloading wave data...")
        progress_dlg.Pulse()
        download_data(gids, pids, ys, months)
        progress_dlg.Update(100)
        wx.MessageBox("Done!", style=wx.OK | wx.CENTRE | wx.STAY_ON_TOP)

    def on_winter(self, event):
        """TO DO"""
        ms = MONTHS["Winter"]
        self.download_data(ms)

    def on_summer(self, event):
        """TO DO"""
        ms = MONTHS["Summer"]
        self.download_data(ms)

    def on_spring(self, event):
        """TO DO"""
        ms = MONTHS["Spring"]
        self.download_data(ms)

    def on_fall(self, event):
        """TO DO"""
        ms = MONTHS["Fall"]
        self.download_data(ms)

    def on_whole(self, event):
        """TO DO"""
        ms = MONTHS["All"]
        self.download_data(ms)

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


class FrameDownloadWind(wx.Frame):
    """
    TO DO
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameDownloadWind, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Welcome to Wind Download Module")
        txt_grids = wx.StaticText(pnl, label="Grid IDs (comma-separated):")
        txt_years = wx.StaticText(pnl, label="Year range (comma-separated):")

        # put a text entry for grid IDs
        self.ent_grids = wx.TextCtrl(pnl, value="wna,at_10m")
        self.ent_years = wx.TextCtrl(pnl, value="1999,2018")

        # put some buttons
        btn_winter = wx.Button(pnl, label="Winter")
        btn_summer = wx.Button(pnl, label="Summer")
        btn_spring = wx.Button(pnl, label="Spring")
        btn_fall = wx.Button(pnl, label="Fall")
        btn_whole = wx.Button(pnl, label="Whole year")
        btn_exit = wx.Button(pnl, label="Exit")

        # associate a handler function to the buttons
        btn_winter.Bind(wx.EVT_BUTTON, self.on_winter)
        btn_summer.Bind(wx.EVT_BUTTON, self.on_summer)
        btn_spring.Bind(wx.EVT_BUTTON, self.on_spring)
        btn_fall.Bind(wx.EVT_BUTTON, self.on_fall)
        btn_whole.Bind(wx.EVT_BUTTON, self.on_whole)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        ### TO DO ###
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(txt_grids, sizer_flags)
        sizer.Add(self.ent_grids, sizer_flags)
        sizer.Add(txt_years, sizer_flags)
        sizer.Add(self.ent_years, sizer_flags)
        sizer.Add(btn_winter, sizer_flags)
        sizer.Add(btn_summer, sizer_flags)
        sizer.Add(btn_spring, sizer_flags)
        sizer.Add(btn_fall, sizer_flags)
        sizer.Add(btn_whole, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def download_data(self, months):
        """Download data for the given months."""
        gids = self.ent_grids.GetLineText(0).split(",")
        pids = ["wind"]
        yi, yf = self.ent_years.GetLineText(0).split(",")
        ys = [*range(int(yi), int(yf) + 1)]
        progress_dlg = wx.ProgressDialog(
            "Download", "Downloading wind data...")
        progress_dlg.Pulse()
        download_data(gids, pids, ys, months)
        wx.MessageBox("Done!", style=wx.OK | wx.CENTRE | wx.STAY_ON_TOP)

    def on_winter(self, event):
        """TO DO"""
        ms = MONTHS["Winter"]
        self.download_data(ms)

    def on_summer(self, event):
        """TO DO"""
        ms = MONTHS["Summer"]
        self.download_data(ms)

    def on_spring(self, event):
        """TO DO"""
        ms = MONTHS["Spring"]
        self.download_data(ms)

    def on_fall(self, event):
        """TO DO"""
        ms = MONTHS["Fall"]
        self.download_data(ms)

    def on_whole(self, event):
        """TO DO"""
        ms = MONTHS["All"]
        self.download_data(ms)

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


class FrameDownload(wx.Frame):
    """
    A frame that asks for parameter to download climate data.
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameDownload, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Welcome to Download Database Information Module")

        # put some buttons
        btn_waves = wx.Button(pnl, label="Wave Information Extraction")
        btn_wind = wx.Button(pnl, label="Wind Information Extraction")
        btn_exit = wx.Button(pnl, label="Exit")

        # associate a handler function to the buttons
        btn_waves.Bind(wx.EVT_BUTTON, self.on_waves)
        btn_wind.Bind(wx.EVT_BUTTON, self.on_wind)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(btn_waves, sizer_flags)
        sizer.Add(btn_wind, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def on_waves(self, event):
        """Hides the main download frame and opens the waves download frame."""
        self.Close(True)
        frm_waves = FrameDownloadWaves(
            None, title="Wave Information Extraction")
        frm_waves.Show()

    def on_wind(self, event):
        """Hides the main download frame and opens the wind download frame."""
        self.Close(True)
        frm_wind = FrameDownloadWind(None, title="Wind Information Extraction")
        frm_wind.Show()

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


class FrameAnalysisShortTerm(wx.Frame):
    """
    TO DO
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAnalysisShortTerm, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Welcome to Analysis and Processing of Climate Information Module")

        # put some buttons
        btn_height_exceedance = wx.Button(pnl, label="Exceedance Probability of Hs (m)")
        btn_period_exceedance = wx.Button(pnl, label="Exceedance Probability of Tp (s)")
        btn_height_joint = wx.Button(pnl, label="Joint Probability of Hs (m) - θ (deg)")
        btn_roses = wx.Button(pnl, label="Wave Roses")
        btn_exit = wx.Button(pnl, label="Exit")

        # associate a handler function to the buttons
        btn_height_exceedance.Bind(wx.EVT_BUTTON, self.on_height_exceedance)
        btn_period_exceedance.Bind(wx.EVT_BUTTON, self.on_period_exceedance)
        btn_height_joint.Bind(wx.EVT_BUTTON, self.on_height_joint)
        btn_roses.Bind(wx.EVT_BUTTON, self.on_roses)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        grdsizer = wx.GridSizer(rows=2, cols=2, vgap=5, hgap=5)
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(grdsizer, sizer_flags)
        grdsizer.Add(btn_height_exceedance, sizer_flags)
        grdsizer.Add(btn_period_exceedance, sizer_flags)
        grdsizer.Add(btn_height_joint, sizer_flags)
        grdsizer.Add(btn_roses, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def on_height_exceedance(self, event):
        """Plots Probability of Exceedance Estimates of mean Significant Wave Height for a season."""
        season = wx.GetSingleChoice("Select a season to analyze:", "Select season", ["Winter", "Summer", "Spring", "Fall"])
        if season:
            ms = MONTHS[season]
            str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value="-74.85,11.11")
            str_lon, str_lat = str_coords.split(",")
            lon = float(str_lon)
            lat = float(str_lat)
            if lon < 0:
                lon = 360 + lon
            progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
            progress_dlg.Pulse()
            fname = f"tmp/hs-{season}-{lon}-{lat}.tmp"
            try:
                x, p = load_obj(fname)
            except FileNotFoundError:
                data = frequency_curve("hs", ms, lon, lat)
                save_obj(data, fname)
                x, p = data
            progress_dlg.Update(100)
            frm_canvas = FrameCanvas(None, title="Probability of Exceedance for Significant Wave Height")
            ax = frm_canvas.fig.add_subplot()
            ax.semilogy(x, p, "o")
            ax.set_xlabel("Significant Wave Height (m)")
            ax.set_ylabel("Probability of Exceedance")
            ax.set_title(season)
            ax.grid(True)
            frm_canvas.Show()

    def on_period_exceedance(self, event):
        """Plots Probability of Exceedance Estimates of mean peak period for a season."""
        season = wx.GetSingleChoice("Select a season to analyze:", "Select season", ["Winter", "Summer", "Spring", "Fall"])
        if season:
            ms = MONTHS[season]
            str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value="-74.85,11.11")
            str_lon, str_lat = str_coords.split(",")
            lon = float(str_lon)
            lat = float(str_lat)
            if lon < 0:
                lon = 360 + lon
            progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
            progress_dlg.Pulse()
            fname = f"tmp/tp-{season}-{lon}-{lat}.tmp"
            try:
                x, p = load_obj(fname)
            except FileNotFoundError:
                data = frequency_curve("tp", ms, lon, lat)
                save_obj(data, fname)
                x, p = data
            progress_dlg.Update(100)
            frm_canvas = FrameCanvas(None, title="Probability of Exceedance for Peak Period")
            ax = frm_canvas.fig.add_subplot()
            ax.semilogy(x, p, "o")
            ax.set_xlabel("Peak Period (s)")
            ax.set_ylabel("Probability of Exceedance")
            ax.set_title(season)
            ax.grid(True)
            frm_canvas.Show()

    def on_height_joint(self, event):
        """TO DO"""
        season = wx.GetSingleChoice("Select a season to analyze:", "Select season", ["Winter", "Summer", "Spring", "Fall"])
        if season:
            ms = MONTHS[season]
            str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value="-74.85,11.11")
            str_lon, str_lat = str_coords.split(",")
            lon = float(str_lon)
            lat = float(str_lat)
            if lon < 0:
                lon = 360 + lon
            progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
            progress_dlg.Pulse()
            fname = f"tmp/dp-hs-{season}-{lon}-{lat}.tmp"
            try:
                dp_hs = load_obj(fname)
            except FileNotFoundError:
                dp_hs  = joint_distribution(["dp", "hs"], ms, lon, lat)
                save_obj(dp_hs, fname)
            progress_dlg.Update(100)
            frm_canvas = FrameCanvas(None, title="Joint Probability of Hs - θ")
            ax = frm_canvas.fig.add_subplot()
            kernel = stats.gaussian_kde(dp_hs)
            dp, hs = dp_hs
            dp, hs = np.meshgrid(
                np.linspace(dp.min(), dp.max(), N),
                np.linspace(hs.min(), hs.max(), N//2)
            )
            p = np.reshape(
                kernel(np.vstack([dp.flatten(), hs.flatten()])),
                (N//2, N)
            )
            im = ax.imshow(p, origin="lower", extent=(0, 360, hs.min(), hs.max()), aspect="auto", cmap="jet")
            cbar = frm_canvas.fig.colorbar(im)
            cbar.ax.set_ylabel("Probability")
            cs = ax.contour(dp, hs, p, colors="k", levels=4, linewidths=1)
            ax.clabel(cs, inline_spacing=0.1)
            ax.set_xlabel("Average direction at the peak period (deg)")
            ax.set_ylabel("Significant Wave Height (m)")
            ax.set_title(season)
            ax.grid(True)
            frm_canvas.Show()

    def on_roses(self, event):
        """TO DO"""
        season = wx.GetSingleChoice("Select a season to analyze:", "Select season", ["Winter", "Summer", "Spring", "Fall"])
        if season:
            ms = MONTHS[season]
            str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value="-74.85,11.11")
            str_lon, str_lat = str_coords.split(",")
            lon = float(str_lon)
            lat = float(str_lat)
            if lon < 0:
                lon = 360 + lon
            progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
            progress_dlg.Pulse()
            fname = f"tmp/dp-hs-{season}-{lon}-{lat}.tmp"
            try:
                dp_hs = load_obj(fname)
            except FileNotFoundError:
                dp_hs  = joint_distribution(["dp", "hs"], ms, lon, lat)
                save_obj(dp_hs, fname)
            progress_dlg.Update(100)
            frm_canvas = FrameCanvas(None, title="Wave Rose")
            ax = frm_canvas.fig.add_subplot(projection="windrose")
            dp, hs = dp_hs
            ax.bar(dp, hs, normed=True)
            ax.set_title("Significant Wave Height Rose")
            ax.set_legend(title="$H_s$ (m)")
            frm_canvas.Show()


    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)

class FrameAnalysisLongTermStorm(wx.Frame):
    """
    TO DO
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAnalysisLongTermStorm, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Welcome to Analysis and Processing of Climate Information Module")

        # put some buttons
        btn_energetic = wx.Button(pnl, label="Energetic Analysis")
        btn_storms_annual = wx.Button(pnl, label="Mean and maximum annual number of storms")
        btn_storms_monthly = wx.Button(pnl, label="Monthly mean and maximum number of storms")
        btn_energies_annual = wx.Button(pnl, label="Annual mean and maximum number of storms with normalized energy")
        btn_energies_monthly = wx.Button(pnl, label="Monthly mean and maximum number of storms with normalized energy")
        btn_exit = wx.Button(pnl, label="Exit")

        # associate a handler function to the buttons
        btn_energetic.Bind(wx.EVT_BUTTON, self.on_energetic)
        btn_storms_annual.Bind(wx.EVT_BUTTON, self.on_storms_annual)
        btn_storms_monthly.Bind(wx.EVT_BUTTON, self.on_storms_monthly)
        btn_energies_annual.Bind(wx.EVT_BUTTON, self.on_energies_annual)
        btn_energies_monthly.Bind(wx.EVT_BUTTON, self.on_energies_monthly)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(btn_energetic, sizer_flags)
        sizer.Add(btn_storms_annual, sizer_flags)
        sizer.Add(btn_storms_monthly, sizer_flags)
        sizer.Add(btn_energies_annual, sizer_flags)
        sizer.Add(btn_energies_monthly, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def on_energetic(self, event):
        """To DO"""
        ms = MONTHS["All"]
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value="-74.85,11.11")
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        fname = "tmp/dp-hs-tp-all-" + f"{lon}-{lat}.tmp"
        try:
            data = load_obj(fname)
        except FileNotFoundError:
            data = load_data(["dp", "hs", "tp"], ms, lon, lat)
            save_obj(data, fname)
        progress_dlg.Update(100)
        dp = []
        hs = []
        tp = []
        for k in data["dp"]:
            dp += data["dp"][k]
            hs += data["hs"][k]
            tp += data["tp"][k]
        dp = np.array(dp)
        hs = np.array(hs)
        tp = np.array(tp)
        p97_hs = np.percentile(hs, 97)
        i = hs >= p97_hs
        hs = hs[i]
        tp = tp[i]
        dp = dp[i]
        frm_canvas = FrameCanvas(None, title="Energetic Analysis")
        frm_canvas.fig.suptitle("Energetic Analysis")
        ax1 = frm_canvas.fig.add_subplot(1, 2, 1, projection="windrose")
        ax1.bar(dp, tp, normed=True)
        ax1.set_title("Wave period rose calculated energetically by storm")
        ax1.set_legend(title="$T_p$ (s)")
        ax2 = frm_canvas.fig.add_subplot(1, 2, 2, projection="windrose")
        en = hs/p97_hs
        ax2.bar(dp, en, normed=True)
        ax2.set_title("Significant wave height rose calculated energetically by storm")
        ax2.set_legend(title="$E$ (-)")
        frm_canvas.Show()
        

    def on_storms_annual(self, event):
        """Analysis of storms ocurrences"""
        ms = MONTHS["All"]
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value="-74.85,11.11")
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        fname = "tmp/dp-hs-tp-all-" + f"{lon}-{lat}.tmp"
        try:
            data = load_obj(fname)
        except FileNotFoundError:
            data = load_data(["dp", "hs", "tp"], ms, lon, lat)
            save_obj(data, fname)
        progress_dlg.Update(100)
        hsall = merge_data(data["hs"])
        p97_hs = np.percentile(hsall, 97)
        p99_hs = np.percentile(hsall, 99)
        hs = {}
        for y, m in data["hs"]:
            if y in hs:
                hs[y].extend(data["hs"][(y, m)])
            else:
                hs[y] = data["hs"][(y, m)]
        n_events_97 = []
        n_events_99 = []
        ys = [*range(min(hs), max(hs) + 1)]
        for y in ys:
            n_events_97.append(sum(x > p97_hs for x in hs[y]))
            n_events_99.append(sum(x > p99_hs for x in hs[y]))
        frm_canvas = FrameCanvas(None, title="Mean and maximum annual number of storms")
        x = np.arange(len(ys))
        frm_canvas.fig.suptitle("Mean and maximum annual number of storms")
        ax1 = frm_canvas.fig.add_subplot(2, 1, 1)
        ax1.bar(x, n_events_97, tick_label=[*map(str, ys)])
        ax1.set_title("$H_s > H_{s,97}$")
        ax1.grid(True)
        ax2 = frm_canvas.fig.add_subplot(2, 1, 2)
        ax2.bar(x, n_events_99, tick_label=[*map(str, ys)])
        ax2.set_title("$H_s > H_{s,99}$")
        ax2.grid(True)
        frm_canvas.Show()

    def on_storms_monthly(self, event):
        """TO DO"""
        ms = MONTHS["All"]
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value="-74.85,11.11")
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        fname = "tmp/dp-hs-tp-all-" + f"{lon}-{lat}.tmp"
        try:
            data = load_obj(fname)
        except FileNotFoundError:
            data = load_data(["dp", "hs", "tp"], ms, lon, lat)
            save_obj(data, fname)
        progress_dlg.Update(100)
        hsall = merge_data(data["hs"])
        p97_hs = np.percentile(hsall, 97)
        p99_hs = np.percentile(hsall, 99)
        hs = {}
        for y, m in data["hs"]:
            if m in hs:
                hs[m].extend(data["hs"][(y, m)])
            else:
                hs[m] = data["hs"][(y, m)]
        n_events_97 = []
        n_events_99 = []
        for m in ms:
            n_events_97.append(sum(x > p97_hs for x in hs[m]))
            n_events_99.append(sum(x > p99_hs for x in hs[m]))
        frm_canvas = FrameCanvas(None, title="Monthly mean and maximum number of storms")
        x = np.arange(12)
        frm_canvas.fig.suptitle("Monthly mean and maximum number of storms")
        ax1 = frm_canvas.fig.add_subplot(2, 1, 1)
        ax1.bar(x, n_events_97, tick_label=STR_MONTHS)
        ax1.grid(True)
        ax1.set_title("$H_s > H_{s,97}$")
        ax2 = frm_canvas.fig.add_subplot(2, 1, 2)
        ax2.bar(x, n_events_99, tick_label=STR_MONTHS)
        ax2.set_title("$H_s > H_{s,99}$")
        ax2.grid(True)
        frm_canvas.Show()


    def on_energies_annual(self, event):
        """Analysis of storms ocurrences"""
        ms = MONTHS["All"]
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value="-74.85,11.11")
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        fname = "tmp/dp-hs-tp-all-" + f"{lon}-{lat}.tmp"
        try:
            data = load_obj(fname)
        except FileNotFoundError:
            data = load_data(["dp", "hs", "tp"], ms, lon, lat)
            save_obj(data, fname)
        progress_dlg.Update(100)
        hsall = merge_data(data["hs"])
        p97_hs = np.percentile(hsall, 97)
        p99_hs = np.percentile(hsall, 99)
        hs = {}
        for y, m in data["hs"]:
            if y in hs:
                hs[y].extend(data["hs"][(y, m)])
            else:
                hs[y] = data["hs"][(y, m)]
        n_events_97 = []
        n_events_99 = []
        ys = [*range(min(hs), max(hs) + 1)]
        for y in ys:
            n_events_97.append(sum(h/p97_hs > 1 for h in hs[y]))
            n_events_99.append(sum(h/p97_hs > p99_hs/p97_hs for h in hs[y]))
        frm_canvas = FrameCanvas(None, title="Mean and maximum annual number of storms with normalized energy")
        x = np.arange(len(ys))
        frm_canvas.fig.suptitle("Mean and maximum annual number of storms with normalized energy")
        ax1 = frm_canvas.fig.add_subplot(2, 1, 1)
        ax1.bar(x, n_events_97, tick_label=[*map(str, ys)])
        ax1.set_title("$E > 1$")
        ax1.grid(True)
        ax2 = frm_canvas.fig.add_subplot(2, 1, 2)
        ax2.bar(x, n_events_99, tick_label=[*map(str, ys)])
        ax2.set_title(f"$E > {p99_hs/p97_hs:3.1f}$")
        ax2.grid(True)
        frm_canvas.Show()

    def on_energies_monthly(self, event):
        """TO DO"""
        ms = MONTHS["All"]
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value="-74.85,11.11")
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        fname = "tmp/dp-hs-tp-all-" + f"{lon}-{lat}.tmp"
        try:
            data = load_obj(fname)
        except FileNotFoundError:
            data = load_data(["dp", "hs", "tp"], ms, lon, lat)
            save_obj(data, fname)
        progress_dlg.Update(100)
        hsall = merge_data(data["hs"])
        p97_hs = np.percentile(hsall, 97)
        p99_hs = np.percentile(hsall, 99)
        hs = {}
        for y, m in data["hs"]:
            if m in hs:
                hs[m].extend(data["hs"][(y, m)])
            else:
                hs[m] = data["hs"][(y, m)]
        n_events_97 = []
        n_events_99 = []
        for m in ms:
            n_events_97.append(sum(h/p97_hs > 1 for h in hs[m]))
            n_events_99.append(sum(h/p97_hs > p99_hs/p97_hs for h in hs[m]))
        frm_canvas = FrameCanvas(None, title="Monthly mean and maximum number of storms with normalized energy")
        x = np.arange(12)
        frm_canvas.fig.suptitle("Monthly mean and maximum number of storms with normalized energy")
        ax1 = frm_canvas.fig.add_subplot(2, 1, 1)
        ax1.bar(x, n_events_97, tick_label=STR_MONTHS)
        ax1.grid(True)
        ax1.set_title("$E > 1$")
        ax2 = frm_canvas.fig.add_subplot(2, 1, 2)
        ax2.bar(x, n_events_99, tick_label=STR_MONTHS)
        ax2.set_title(f"$E > {p99_hs/p97_hs:3.1f}$")
        ax2.grid(True)
        frm_canvas.Show()

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)

class FrameAnalysisLongTermMaps(wx.Frame):
    """
    TO DO
    """

    def __init__(self, pr, q, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAnalysisLongTermMaps, self).__init__(*args, **kw)

        # create a panel in the frame
        self.pr = pr
        self.q = q
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Welcome to Analysis and Processing of Climate Information Module")

        # put some buttons
        btn_waves = wx.Button(pnl, label="Waves")
        btn_wind = wx.Button(pnl, label="Wind")
        btn_exit = wx.Button(pnl, label="Exit")

        # associate a handler function to the buttons
        btn_waves.Bind(wx.EVT_BUTTON, self.on_waves)
        btn_wind.Bind(wx.EVT_BUTTON, self.on_wind)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        grdsizer = wx.GridSizer(rows=1, cols=2, vgap=5, hgap=5)
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(grdsizer, sizer_flags)
        grdsizer.Add(btn_waves, sizer_flags)
        grdsizer.Add(btn_wind, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def on_waves(self, event):
        """TO DO"""
        str_coords = wx.GetTextFromUser("Coordinates (lon1, lon2, lat1, lat2):", default_value="-75.3,-74.1,10,11.5")
        lon1, lon2, lat1, lat2 = [float(x) for x in str_coords.split(",")]
        if lon1 < 0:
            lon1 = 360 + lon1
        if lon2 < 0:
            lon2 = 360 + lon2
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        fname = "tmp/annual-maxima-waves.tmp"
        try:
            maxima = load_obj(fname)
        except FileNotFoundError:
            maxima = load_max("hs")
            save_obj(maxima, fname)
        progress_dlg.Update(100)
        lon = maxima[0].coords["longitude"].values
        lat = maxima[0].coords["latitude"].values
        da_hs_q = xr.DataArray(
            np.nanquantile(np.stack(maxima), self.q, 0),
            dims=["latitude", "longitude"],
            coords={"latitude": lat, "longitude": lon}
        )
        data = da_hs_q.sel(latitude=slice(lat2, lat1), longitude=slice(lon1, lon2))
        min_lon = data.coords["longitude"].values.min()
        max_lon = data.coords["longitude"].values.max()
        if min_lon > 180:
            min_lon -= 360
        if max_lon > 180:
            max_lon -= 360
        min_lat = data.coords["latitude"].values.min()
        max_lat = data.coords["latitude"].values.max()
        extent = (min_lon, max_lon, min_lat, max_lat)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title(f"Return period (Annual) = {self.pr} years")
        data = interp_idw(interp_idw(data))
        im = ax.imshow(data, origin="upper", extent=extent, transform=ccrs.PlateCarree(), cmap="jet", interpolation="bilinear")
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel("Significant Wave Height (m)")
        ax.add_feature(LAND_10M)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        plt.tight_layout()
        plt.show()


    def on_wind(self, event):
        """TO DO"""
        str_coords = wx.GetTextFromUser("Coordinates (lon1, lon2, lat1, lat2):", default_value="-75.3,-74.1,10,11.5")
        lon1, lon2, lat1, lat2 = [float(x) for x in str_coords.split(",")]
        if lon1 < 0:
            lon1 = 360 + lon1
        if lon2 < 0:
            lon2 = 360 + lon2
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wind data...")
        progress_dlg.Pulse()
        fname = "tmp/annual-maxima-wind.tmp"
        try:
            maxima = load_obj(fname)
        except FileNotFoundError:
            maxima = load_max("wind")
            save_obj(maxima, fname)
        progress_dlg.Update(100)
        lon = maxima[0].coords["longitude"].values
        lat = maxima[0].coords["latitude"].values
        da_wind_q = xr.DataArray(
            np.nanquantile(np.stack(maxima), self.q, 0),
            dims=["latitude", "longitude"],
            coords={"latitude": lat, "longitude": lon}
        )
        data = da_wind_q.sel(latitude=slice(lat2, lat1), longitude=slice(lon1, lon2))
        min_lon = data.coords["longitude"].values.min()
        max_lon = data.coords["longitude"].values.max()
        if min_lon > 180:
            min_lon -= 360
        if max_lon > 180:
            max_lon -= 360
        min_lat = data.coords["latitude"].values.min()
        max_lat = data.coords["latitude"].values.max()
        extent = (min_lon, max_lon, min_lat, max_lat)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title(f"Return period (Annual) = {self.pr} years")
        data = interp_idw(interp_idw(data))
        im = ax.imshow(data, origin="upper", extent=extent, transform=ccrs.PlateCarree(), cmap="jet", interpolation="bilinear")
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel("Wind speed (m/s)")
        ax.add_feature(LAND_10M)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        plt.tight_layout()
        plt.show()

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


class FrameAnalysisLongTerm(wx.Frame):
    """
    TO DO
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAnalysisLongTerm, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Welcome to Analysis and Processing of Climate Information Module")

        # put some buttons
        btn_weibull = wx.Button(pnl, label="Long term though Weibull Distribution")
        btn_storms = wx.Button(pnl, label="Ocurrences of storms analysis")
        btn_maps = wx.Button(pnl, label="Regional maps")
        btn_exit = wx.Button(pnl, label="Exit")

        # associate a handler function to the buttons
        btn_weibull.Bind(wx.EVT_BUTTON, self.on_weibull)
        btn_storms.Bind(wx.EVT_BUTTON, self.on_storms)
        btn_maps.Bind(wx.EVT_BUTTON, self.on_maps)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        grdsizer = wx.GridSizer(rows=2, cols=2, vgap=5, hgap=5)
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(grdsizer, sizer_flags)
        grdsizer.Add(btn_weibull, sizer_flags)
        grdsizer.Add(btn_storms, sizer_flags)
        grdsizer.Add(btn_maps, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def on_weibull(self, event):
        """Extreme value analysis - Maximum Weibull distribution"""
        season = wx.GetSingleChoice("Select a season to analyze:", "Select season", ["Winter", "Summer", "Spring", "Fall"])
        if season:
            ms = MONTHS[season]
            str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value="-74.85,11.11")
            str_lon, str_lat = str_coords.split(",")
            lon = float(str_lon)
            lat = float(str_lat)
            if lon < 0:
                lon = 360 + lon
            progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
            progress_dlg.Pulse()
            fname = f"tmp/hs-peaks-{season}-{lon}-{lat}.tmp"
            try:
                peaks_hs = load_obj(fname)
            except FileNotFoundError:
                peaks_hs = weibull_data("hs", ms, lon, lat)
                save_obj(peaks_hs, fname)
            progress_dlg.Update(100)
            analysis = weibull.Analysis(peaks_hs, unit="m")
            analysis.fit()
            analysis.probplot()

    def on_maps(self, event):
        """TO DO"""
        self.Close(True)
        pr = wx.GetTextFromUser("Return period (years):", default_value="100")
        q = 1 - 1/float(pr)
        frm = FrameAnalysisLongTermMaps(pr, q, None, title="Regional Maps")
        frm.Show()

    def on_storms(self, event):
        """Analysis of storms ocurrences"""
        self.Close(True)
        frm = FrameAnalysisLongTermStorm(None, title="Storm Analysis")
        frm.Show()

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


class FrameAnalysis(wx.Frame):
    """
    A frame that asks for short-term or long-term type of analysis.
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAnalysis, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Analysis and Processing of Climate Information")

        # put some buttons
        btn_short_term = wx.Button(pnl, label="Short Term Analysis")
        btn_long_term = wx.Button(pnl, label="Long Term Analysis")
        btn_exit = wx.Button(pnl, label="Exit")

        # associate a handler function to the buttons
        btn_short_term.Bind(wx.EVT_BUTTON, self.on_short_term)
        btn_long_term.Bind(wx.EVT_BUTTON, self.on_long_term)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(btn_short_term, sizer_flags)
        sizer.Add(btn_long_term, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def on_short_term(self, event):
        """TO DO"""
        self.Close(True)
        frm_short_term = FrameAnalysisShortTerm(None, title="Short Term Analysis")
        frm_short_term.Show()

    def on_long_term(self, event):
        """TO DO"""
        self.Close(True)
        frm_long_term = FrameAnalysisLongTerm(None, title="Long Term Analysis")
        frm_long_term.Show()

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


class FrameMain(wx.Frame):
    """
    A Frame that shows options for downloading and analysis of climate data.
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameMain, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(pnl, label="Welcome to APPMAR 1.0")
        txt_author = wx.StaticText(
            pnl,
            label="Marianella Bolívar\nDiego Casas\nGerman Rivillas Ospina, PhD",
            style=wx.ALIGN_CENTER,
        )

        # put some buttons
        btn_download = wx.Button(pnl, label="Download Database Information")
        btn_analysis = wx.Button(
            pnl, label="Analysis and Processing of Climate Information")
        btn_exit = wx.Button(pnl, label="Exit")

        # associate a handler function to the buttons
        btn_download.Bind(wx.EVT_BUTTON, self.on_download)
        btn_analysis.Bind(wx.EVT_BUTTON, self.on_analysis)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(txt_author, sizer_flags)
        sizer.Add(btn_download, sizer_flags)
        sizer.Add(btn_analysis, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def on_download(self, event):
        """Hides the main frame and opens the download frame."""
        frm_download = FrameDownload(
            None, title="Download Database Information")
        frm_download.Show()

    def on_analysis(self, event):
        """Hides the main frame and opens the analysis frame."""
        os.makedirs("tmp", exist_ok=True)
        frm_analysis = FrameAnalysis(None, title="Analysis and Processing of Climate Information Module")
        frm_analysis.Show()

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


if __name__ == "__main__":
    APP = wx.App()
    FRM = FrameMain(None, title="APPMAR 1.0")
    FRM.Show()
    APP.MainLoop()
