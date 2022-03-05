"""
APPMAR 1.0
A toolbox for management of meteorological and marine data on limited information regions

Marianella Bolívar
Diego Casas
Germán Rivillas Ospina, PhD
"""

import os
import wx
import wx.lib.agw.hyperlink as hl
import matplotlib
from libappmar import download_data, frequency_curve, joint_distribution, load_data, load_obj, save_obj, merge_data, weibull_data, load_max, interp_idw, get_defaults, plot_weibull, compute_clusters, plot_clusters, kernel_min_max, roseplot, format_title
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as mtick
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xarray as xr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

MONTHS, DEFAULT_COORD, BOX = get_defaults()

LAND_10M = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor="k", facecolor="grey")
N = 181

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams['mathtext.fontset'] = "custom"
matplotlib.rcParams['mathtext.rm'] = "Times New Roman"
matplotlib.rcParams['mathtext.it'] = "Times New Roman:italic"
matplotlib.rcParams['mathtext.bf'] = "Times New Roman:bold"

STR_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

class FrameCanvas(wx.Frame):
    def __init__(self, figsize=(6.5, 4), *args, **kw):
        super(FrameCanvas, self).__init__(*args, **kw)

        self.fig = Figure(figsize=figsize)
        self.canvas = FigureCanvas(self, -1, self.fig)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        self.SetSizer(self.sizer)

        self.add_toolbar()  # comment this out for no toolbar
        self.Fit()

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
            pnl, label="Wave Download Module")
        txt_welcome.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.NORMAL))
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
            pnl, label="Wind Download Module")
        txt_welcome.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.NORMAL))
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
            pnl, label="Module – Download Database Information")
        txt_welcome.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.NORMAL))

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


class FrameAnalysisMeanClimate(wx.Frame):
    """
    TO DO
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAnalysisMeanClimate, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Module – Analysis and Processing of Climate Information")
        txt_welcome.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.NORMAL))

        # put some buttons
        btn_height_exceedance = wx.Button(pnl, label="Exceedance Probability of Hs (m)")
        btn_period_exceedance = wx.Button(pnl, label="Exceedance Probability of Tp (s)")
        btn_height_joint = wx.Button(pnl, label="Joint Probability of Hs (m) - θ (deg)")
        btn_roses = wx.Button(pnl, label="Wave Roses")
        btn_clusters = wx.Button(pnl, label="Representative Hs (m) - Tp (s) scenarios")
        btn_exit = wx.Button(pnl, label="Exit")

        # global limits checkbox
        self.chk_globlims = wx.CheckBox(pnl, label="Global limits for plots")

        # associate a handler function to the buttons
        btn_height_exceedance.Bind(wx.EVT_BUTTON, self.on_height_exceedance)
        btn_period_exceedance.Bind(wx.EVT_BUTTON, self.on_period_exceedance)
        btn_height_joint.Bind(wx.EVT_BUTTON, self.on_height_joint)
        btn_roses.Bind(wx.EVT_BUTTON, self.on_roses)
        btn_clusters.Bind(wx.EVT_BUTTON, self.on_clusters)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        grdsizer = wx.GridSizer(rows=3, cols=2, vgap=5, hgap=5)
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(grdsizer, sizer_flags)
        sizer.Add(self.chk_globlims, sizer_flags)
        grdsizer.Add(btn_height_exceedance, sizer_flags)
        grdsizer.Add(btn_period_exceedance, sizer_flags)
        grdsizer.Add(btn_height_joint, sizer_flags)
        grdsizer.Add(btn_roses, sizer_flags)
        grdsizer.Add(btn_clusters, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def on_height_exceedance(self, event):
        """Plots Probability of Exceedance Estimates of mean Significant Wave Height for a season."""
        season = wx.GetSingleChoice("Select a season to analyze:", "Select season", ["Winter", "Summer", "Spring", "Fall", "Combined", "All"])
        if season:
            seasons = []
            if season == "Combined":
                lbls = ["Winter", "Summer", "Spring", "Fall"]
                for s in lbls:
                    seasons.append(MONTHS[s])
            else:
                lbls = [None]
                seasons.append(MONTHS[season])
            str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
            str_lon, str_lat = str_coords.split(",")
            lon = float(str_lon)
            lat = float(str_lat)
            if lon < 0:
                lon = 360 + lon
            progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
            progress_dlg.Pulse()
            xs = []
            ps = []
            for ms in seasons:
                data = frequency_curve("hs", ms, lon, lat)
                x, p = data
                xs.append(x)
                ps.append(p)
            progress_dlg.Update(100)
            frm_canvas = FrameCanvas(parent=None, title="Probability of Exceedance for Significant Wave Height")
            ax = frm_canvas.fig.add_subplot()
            for x, p, lbl in zip(xs, ps, lbls):
                ax.scatter(x, p*100, marker=".", s=8, label=lbl)
            if season == "Combined":
                lgnd = ax.legend()
                for handle in lgnd.legendHandles:
                    handle.set_sizes([50])
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_xlabel("Significant Wave Height (m)")
            ax.set_ylabel("Probability of Exceedance")
            ax.set_title(format_title(lon, lat, str_coords=str_coords, season=season))
            if self.chk_globlims.IsChecked():
                arr = frequency_curve("hs", MONTHS["All"], lon, lat)[0]
                ax.set_xlim((arr.min(), arr.max()))
            ax.grid(True)
            frm_canvas.Show()

    def on_period_exceedance(self, event):
        """Plots Probability of Exceedance Estimates of mean peak period for a season."""
        season = wx.GetSingleChoice("Select a season to analyze:", "Select season", ["Winter", "Summer", "Spring", "Fall", "Combined", "All"])
        if season:
            seasons = []
            if season == "Combined":
                lbls = ["Winter", "Summer", "Spring", "Fall"]
                for s in lbls:
                    seasons.append(MONTHS[s])
            else:
                lbls = [None]
                seasons.append(MONTHS[season])
            str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
            str_lon, str_lat = str_coords.split(",")
            lon = float(str_lon)
            lat = float(str_lat)
            if lon < 0:
                lon = 360 + lon
            progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
            progress_dlg.Pulse()
            xs = []
            ps = []
            for ms in seasons:
                data = frequency_curve("tp", ms, lon, lat)
                x, p = data
                xs.append(x)
                ps.append(p)
            progress_dlg.Update(100)
            frm_canvas = FrameCanvas(parent=None, title="Probability of Exceedance for Peak Period")
            ax = frm_canvas.fig.add_subplot()
            for x, p, lbl in zip(xs, ps, lbls):
                ax.scatter(x, p*100, marker=".", s=8, label=lbl)
            if season == "Combined":
                lgnd = ax.legend()
                for handle in lgnd.legendHandles:
                    handle.set_sizes([50])
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_xlabel("Peak Period (s)")
            ax.set_ylabel("Probability of Exceedance")
            ax.set_title(format_title(lon, lat, str_coords=str_coords, season=season))
            if self.chk_globlims.IsChecked():
                arr = frequency_curve("tp", MONTHS["All"], lon, lat)[0]
                ax.set_xlim((arr.min(), arr.max()))
            ax.grid(True)
            frm_canvas.Show()

    def on_height_joint(self, event):
        season = wx.GetSingleChoice("Select a season to analyze:", "Select season", ["Winter", "Summer", "Spring", "Fall", "All"])
        if season:
            ms = MONTHS[season]
            str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
            str_lon, str_lat = str_coords.split(",")
            lon = float(str_lon)
            lat = float(str_lat)
            if lon < 0:
                lon = 360 + lon
            progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
            progress_dlg.Pulse()
            dp_hs  = joint_distribution(["dp", "hs"], ms, lon, lat)
            progress_dlg.Update(100)
            frm_canvas = FrameCanvas(parent=None, title="Joint Probability of Hs - θ")
            ax = frm_canvas.fig.add_subplot()
            kernel = stats.gaussian_kde(dp_hs)
            dp, hs = dp_hs
            if self.chk_globlims.IsChecked():
                vmin, vmax, hsmin, hsmax = kernel_min_max(["dp", "hs"], MONTHS, lon, lat, N)
            else:
                vmin = None
                vmax = None
                hsmin = hs.min()
                hsmax = hs.max()
            dp, hs = np.meshgrid(
                np.linspace(0, 360, N),
                np.linspace(hsmin, hsmax, N//2)
            )
            p = np.reshape(
                kernel(np.vstack([dp.flatten(), hs.flatten()])),
                (N//2, N)
            )
            im = ax.imshow(p, origin="lower", extent=(0, 360, hsmin, hsmax), aspect="auto", cmap="GnBu", vmin=vmin, vmax=vmax)
            cbar = frm_canvas.fig.colorbar(im)
            cbar.ax.set_ylabel("Probability density")
            #cs = ax.contour(dp, hs, p, colors="k", levels=4, linewidths=1)
            #ax.clabel(cs, inline_spacing=0.1)
            ax.set_xlabel("Average direction at the peak period (deg)")
            ax.set_ylabel("Significant Wave Height (m)")
            ax.set_title(format_title(lon, lat, str_coords=str_coords, season=season))
            ax.grid(True)
            frm_canvas.Show()

    def on_roses(self, event):
        season = wx.GetSingleChoice("Select a season to analyze:", "Select season", ["Winter", "Summer", "Spring", "Fall", "All"])
        if season:
            ms = MONTHS[season]
            str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
            str_lon, str_lat = str_coords.split(",")
            lon = float(str_lon)
            lat = float(str_lat)
            if lon < 0:
                lon = 360 + lon
            progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
            progress_dlg.Pulse()
            dp_hs  = joint_distribution(["dp", "hs"], ms, lon, lat)
            progress_dlg.Update(100)
            frm_canvas = FrameCanvas(parent=None, title="Wave Rose")
            ax = frm_canvas.fig.add_subplot(projection="polar")
            dp, hs = dp_hs
            if self.chk_globlims.IsChecked():
                arr  = joint_distribution(["dp", "hs"], MONTHS["All"], lon, lat)[1]
                bin_edges = np.quantile(arr, np.linspace(0, 1, 6))
                ax.set_ylim((0, 1))
            else:
                bin_edges = np.histogram_bin_edges(hs, 5)
            bin_edges[-1] = np.inf
            roseplot(dp, hs, bin_edges, dirnames=True, xlabel="$H_s$ (m)", ax=ax)
            ax.set_title(format_title(lon, lat, str_coords=str_coords, season=season), pad=20)
            frm_canvas.fig.set_tight_layout(True)
            frm_canvas.Show()


    def on_clusters(self, event):
        season = "All"
        if season:
            ms = MONTHS[season]
            str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
            str_lon, str_lat = str_coords.split(",")
            lon = float(str_lon)
            lat = float(str_lat)
            if lon < 0:
                lon = 360 + lon
            progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
            progress_dlg.Pulse()
            hs_tp  = joint_distribution(["hs", "tp"], ms, lon, lat)
            progress_dlg.Update(100)
            frm_canvas = FrameCanvas(parent=None, title="Representative Hs - Tp scenarios")
            ax = frm_canvas.fig.add_subplot()
            hs, tp = hs_tp
            pairs = np.column_stack((hs, tp))
            centers, labels = compute_clusters(pairs)
            plot_clusters(pairs, centers, labels, ax)
            ax.grid(True)
            ax.set_title(format_title(lon, lat, str_coords=str_coords, season=season))
            frm_canvas.Show()
            scenarios = " ; ".join(f"{h:.2f},{t:.2f}" for h, t in centers)
            dlg = wx.TextEntryDialog(None,"Hs1,Tp1 ; Hs2,Tp2 ; ... ; HsN,TpN", "Representative Hs-Tp scenarios", scenarios)
            dlg.SetSize((600,180))
            dlg.ShowModal()


    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)

class FrameAnalysisExtremeClimateStorm(wx.Frame):
    """
    TO DO
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAnalysisExtremeClimateStorm, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Module – Analysis and Processing of Climate Information")
        txt_welcome.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.NORMAL))

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
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        data = load_data(["dp", "hs", "tp"], ms, lon, lat)
        progress_dlg.Update(100)
        dp = data["dp"][1].values
        hs = data["hs"][1].values
        tp = data["tp"][1].values
        if not (len(dp) == len(hs) == len(tp)):
            raise Exception("Missing or corrupt files: Check the data directory for corrupt files and download again.")
        p97_hs = np.percentile(hs, 97)
        i = hs >= p97_hs
        hs = hs[i]
        tp = tp[i]
        dp = dp[i]
        notnan = ~np.isnan(hs + tp + dp)
        hs = hs[notnan]
        tp = tp[notnan]
        dp = dp[notnan]
        frm_canvas = FrameCanvas(figsize=(9, 4.5), parent=None, title="Energetic Analysis")        
        ax1 = frm_canvas.fig.add_subplot(1, 2, 1, projection="polar")
        bin_edges1 = np.histogram_bin_edges(tp, 5)
        bin_edges1[-1] = np.inf
        roseplot(dp, tp, bin_edges1, dirnames=True, xlabel="$T_p$ (s)", ax=ax1)
        ax1.set_title("Wave period rose for storm events\n" + format_title(lon, lat, str_coords=str_coords), pad=20)
        ax2 = frm_canvas.fig.add_subplot(1, 2, 2, projection="polar")
        en = hs/p97_hs
        bin_edges2 = np.histogram_bin_edges(en, 5)
        bin_edges2[-1] = np.inf
        roseplot(dp, en, bin_edges2, dirnames=True, xlabel="$E$ (-)", ax=ax2)
        ax2.set_title("Normalized energy rose for storm events\n" + format_title(lon, lat, str_coords=str_coords), pad=20)
        frm_canvas.fig.set_tight_layout(True)
        frm_canvas.Show()
        

    def on_storms_annual(self, event):
        """Analysis of storms ocurrences"""
        ms = MONTHS["All"]
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        data = load_data(["dp", "hs", "tp"], ms, lon, lat)
        progress_dlg.Update(100)
        hsall = data["hs"].values
        p97_hs = np.percentile(hsall, 97)
        p99_hs = np.percentile(hsall, 99)
        hs = {}
        ys = data["hs"].index.year.unique().sort_values()
        for y in ys:
            hs[y] = data["hs"][data["hs"].index.year == y][1].values
        n_events_97 = []
        n_events_99 = []
        for y in ys:
            n_events_97.append(sum(x > p97_hs for x in hs[y]))
            n_events_99.append(sum(x > p99_hs for x in hs[y]))
        frm_canvas = FrameCanvas(parent=None, title="Mean and maximum annual number of storms")
        x = np.arange(len(ys))
        ax = frm_canvas.fig.add_subplot()
        ax.bar(x, n_events_97, bottom=n_events_99, tick_label=[*map(str, ys)],
            label=f"$H_s$ > P97 ({p97_hs:.2f} m)")
        ax.bar(x, n_events_99, label=f"$H_s$ > P99 ({p99_hs:.2f} m)")
        ax.legend()
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True)
        frm_canvas.fig.set_tight_layout(True)
        frm_canvas.Show()

    def on_storms_monthly(self, event):
        """TO DO"""
        ms = MONTHS["All"]
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        data = load_data(["dp", "hs", "tp"], ms, lon, lat)
        progress_dlg.Update(100)
        hsall = data["hs"].values
        p97_hs = np.percentile(hsall, 97)
        p99_hs = np.percentile(hsall, 99)
        hs = {}
        ms = data["hs"].index.month.unique().sort_values()
        for m in ms:
            hs[m] = data["hs"][data["hs"].index.month == m][1].values
        n_events_97 = []
        n_events_99 = []
        for m in ms:
            n_events_97.append(sum(x > p97_hs for x in hs[m]))
            n_events_99.append(sum(x > p99_hs for x in hs[m]))
        frm_canvas = FrameCanvas(parent=None, title="Monthly mean and maximum number of storms")
        x = np.arange(12)
        ax = frm_canvas.fig.add_subplot()
        ax.bar(x, n_events_97, bottom=n_events_99, tick_label=STR_MONTHS,
            label=f"$H_s$ > P97 ({p97_hs:.2f} m)")
        ax.bar(x, n_events_99, label=f"$H_s$ > P99 ({p99_hs:.2f} m)")
        ax.legend()
        ax.grid(True)
        frm_canvas.fig.set_tight_layout(True)
        frm_canvas.Show()


    def on_energies_annual(self, event):
        """Analysis of storms ocurrences"""
        ms = MONTHS["All"]
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        data = load_data(["dp", "hs", "tp"], ms, lon, lat)
        progress_dlg.Update(100)
        hsall = data["hs"].values
        p97_hs = np.percentile(hsall, 97)
        p99_hs = np.percentile(hsall, 99)
        hs = {}
        ys = data["hs"].index.year.unique().sort_values()
        for y in ys:
            hs[y] = data["hs"][data["hs"].index.year == y][1].values
        n_events_97 = []
        n_events_99 = []
        for y in ys:
            n_events_97.append(sum(h/p97_hs > 1 for h in hs[y]))
            n_events_99.append(sum(h/p97_hs > p99_hs/p97_hs for h in hs[y]))
        frm_canvas = FrameCanvas(parent=None, title="Mean and maximum annual number of storms with normalized energy")
        x = np.arange(len(ys))
        ax = frm_canvas.fig.add_subplot()
        ax.bar(x, n_events_97, bottom=n_events_99, tick_label=[*map(str, ys)], label="$E > 1$")
        ax.bar(x, n_events_99, label=f"$E > {p99_hs/p97_hs:.2f}$")
        ax.legend()
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True)
        frm_canvas.fig.set_tight_layout(True)
        frm_canvas.Show()

    def on_energies_monthly(self, event):
        """TO DO"""
        ms = MONTHS["All"]
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        data = load_data(["dp", "hs", "tp"], ms, lon, lat)
        progress_dlg.Update(100)
        hsall = data["hs"].values
        p97_hs = np.percentile(hsall, 97)
        p99_hs = np.percentile(hsall, 99)
        hs = {}
        ms = data["hs"].index.month.unique().sort_values()
        for m in ms:
            hs[m] = data["hs"][data["hs"].index.month == m][1].values
        n_events_97 = []
        n_events_99 = []
        for m in ms:
            n_events_97.append(sum(h/p97_hs > 1 for h in hs[m]))
            n_events_99.append(sum(h/p97_hs > p99_hs/p97_hs for h in hs[m]))
        frm_canvas = FrameCanvas(parent=None, title="Monthly mean and maximum number of storms with normalized energy")
        x = np.arange(12)
        ax = frm_canvas.fig.add_subplot()
        ax.bar(x, n_events_97, bottom=n_events_99, tick_label=STR_MONTHS, label="$E > 1$")
        ax.bar(x, n_events_99, label=f"$E > {p99_hs/p97_hs:.2f}$")
        ax.legend()
        ax.grid(True)
        frm_canvas.fig.set_tight_layout(True)
        frm_canvas.Show()

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)

class FrameAnalysisExtremeClimateMaps(wx.Frame):
    """
    TO DO
    """

    def __init__(self, pr, q, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAnalysisExtremeClimateMaps, self).__init__(*args, **kw)

        # create a panel in the frame
        self.pr = pr
        self.q = q
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Module – Analysis and Processing of Climate Information")
        txt_welcome.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.NORMAL))

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
        str_coords = wx.GetTextFromUser("Coordinates (lon1, lon2, lat1, lat2):", default_value=BOX)
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
        frm_canvas = FrameCanvas(parent=None, title="Regional quantile map")
        ax = frm_canvas.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_title(f"Return period (Annual) = {self.pr} years")
        data = interp_idw(interp_idw(data))
        im = ax.imshow(data, origin="upper", extent=extent, transform=ccrs.PlateCarree(), cmap="GnBu", interpolation="bilinear")
        cbar = frm_canvas.fig.colorbar(im)
        cbar.ax.set_ylabel("Significant Wave Height (m)")
        ax.add_feature(LAND_10M)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'rotation': 45}
        frm_canvas.fig.set_tight_layout(True)
        frm_canvas.Show()


    def on_wind(self, event):
        """TO DO"""
        str_coords = wx.GetTextFromUser("Coordinates (lon1, lon2, lat1, lat2):", default_value=BOX)
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
        frm_canvas = FrameCanvas(parent=None, title="Regional quantile map")
        ax = frm_canvas.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_title(f"Return period (Annual) = {self.pr} years")
        data = interp_idw(interp_idw(data))
        im = ax.imshow(data, origin="upper", extent=extent, transform=ccrs.PlateCarree(), cmap="GnBu", interpolation="bilinear")
        cbar = frm_canvas.fig.colorbar(im)
        cbar.ax.set_ylabel("Wind speed (m/s)")
        ax.add_feature(LAND_10M)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'rotation': 45}
        frm_canvas.fig.set_tight_layout(True)
        frm_canvas.Show()

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


class FrameAnalysisExtremeClimate(wx.Frame):
    """
    TO DO
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAnalysisExtremeClimate, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Module – Analysis and Processing of Climate Information")
        txt_welcome.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.NORMAL))

        # put some buttons
        btn_weibull = wx.Button(pnl, label="EVA through Weibull Distribution")
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
        """Extreme value analysis - Weibull Maximum distribution"""
        ms = MONTHS["All"]
        str_coords = wx.GetTextFromUser("Coordinates (lon, lat):", default_value=DEFAULT_COORD)
        str_lon, str_lat = str_coords.split(",")
        lon = float(str_lon)
        lat = float(str_lat)
        if lon < 0:
            lon = 360 + lon
        progress_dlg = wx.ProgressDialog("Read and analyze", "Reading and analyzing wave data...")
        progress_dlg.Pulse()
        peaks_hs = weibull_data("hs", ms, lon, lat)
        progress_dlg.Update(100)
        frm_canvas = FrameCanvas(parent=None, title="Weibull probability plot")
        ax = frm_canvas.fig.add_subplot()
        ax.set_title("Weibull Probability Plot\n" + format_title(lon, lat, str_coords=str_coords))
        plot_weibull(peaks_hs, ax)
        frm_canvas.Show()

    def on_maps(self, event):
        """TO DO"""
        self.Close(True)
        pr = wx.GetTextFromUser("Return period (years):", default_value="100")
        q = 1 - 1/float(pr)
        frm = FrameAnalysisExtremeClimateMaps(pr, q, None, title="Regional Maps")
        frm.Show()

    def on_storms(self, event):
        """Analysis of storms ocurrences"""
        self.Close(True)
        frm = FrameAnalysisExtremeClimateStorm(None, title="Storm Analysis")
        frm.Show()

    def on_exit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


class FrameAnalysis(wx.Frame):
    """
    A frame that asks for mean and extreme climate type of analysis.
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAnalysis, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # put some text
        txt_welcome = wx.StaticText(
            pnl, label="Analysis and Processing of Climate Information")
        txt_welcome.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.NORMAL))

        # put some buttons
        btn_mean_climate = wx.Button(pnl, label="Mean Climate Analysis")
        btn_extreme_climate = wx.Button(pnl, label="Extreme Climate Analysis")
        btn_exit = wx.Button(pnl, label="Exit")

        # associate a handler function to the buttons
        btn_mean_climate.Bind(wx.EVT_BUTTON, self.on_mean_climate)
        btn_extreme_climate.Bind(wx.EVT_BUTTON, self.on_extreme_climate)
        btn_exit.Bind(wx.EVT_BUTTON, self.on_exit)

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(btn_mean_climate, sizer_flags)
        sizer.Add(btn_extreme_climate, sizer_flags)
        sizer.Add(btn_exit, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def on_mean_climate(self, event):
        """TO DO"""
        self.Close(True)
        frm_mean_climate = FrameAnalysisMeanClimate(None, title="Mean Climate Analysis")
        frm_mean_climate.Show()

    def on_extreme_climate(self, event):
        """TO DO"""
        self.Close(True)
        frm_extreme_climate = FrameAnalysisExtremeClimate(None, title="Extreme Climate Analysis")
        frm_extreme_climate.Show()

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
        txt_appname = wx.StaticText(pnl, label="APPMAR 1.0")
        txt_appname.SetFont(wx.Font(15, wx.DEFAULT, wx.NORMAL, wx.BOLD))

        # put some buttons
        btn_download = wx.Button(pnl, label="Download Database Information")
        btn_analysis = wx.Button(
            pnl, label="Analysis and Processing of Climate Information")

        # associate a handler function to the buttons
        btn_download.Bind(wx.EVT_BUTTON, self.on_download)
        btn_analysis.Bind(wx.EVT_BUTTON, self.on_analysis)

        # create a sizer to manage the layout of child widgets
        sizerv = wx.BoxSizer(wx.VERTICAL)
        stbx_d = wx.StaticBox(pnl, wx.ID_ANY, "Download", size=(350, 200))
        stbx_a = wx.StaticBox(pnl, wx.ID_ANY, "Analysis", size=(350, 200))
        sizerd = wx.StaticBoxSizer(stbx_d, wx.VERTICAL)
        sizera = wx.StaticBoxSizer(stbx_a, wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border(wx.ALL, 10).Center()
        sizerv.Add(txt_appname, sizer_flags)
        sizerv.Add(sizerd, sizer_flags)
        sizerv.Add(sizera, sizer_flags)
        sizerd.Add(btn_download, sizer_flags)
        sizera.Add(btn_analysis, sizer_flags)
        pnl.SetSizer(sizerv)
        sizerv.Fit(self)

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


class FrameAbout(wx.Frame):
    """
    A Frame that shows the About window.
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameAbout, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # add CEMAN logo
        logo = wx.StaticBitmap(pnl, wx.ID_ANY, wx.Bitmap("ceman.png", wx.BITMAP_TYPE_ANY))

        # put some text
        txt_appname = wx.StaticText(pnl, label="APPMAR 1.0 by CEMAN", style=wx.ALIGN_CENTER)
        txt_authors = wx.StaticText(
            pnl,
            label="Authors:\n\nMarianella Bolívar\nDiego Casas\nGerman Rivillas Ospina, PhD",
            style=wx.ALIGN_CENTER,
        )
        lnkgh = hl.HyperLinkCtrl(pnl, wx.ID_ANY, "Source code repository", URL="https://github.com/cemanetwork/appmar")

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        sizer.Add(logo, sizer_flags)
        sizer.Add(txt_appname, sizer_flags)
        sizer.Add(txt_authors, sizer_flags)
        sizer.Add(lnkgh, sizer_flags)
        sizer.AddSpacer(20)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

class FrameStart(wx.Frame):
    """
    A Frame that shows the start window.
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(FrameStart, self).__init__(*args, **kw)

        # create a panel in the frame
        pnl = wx.Panel(self)

        # add CEMAN logo
        logo = wx.StaticBitmap(pnl, wx.ID_ANY, wx.Bitmap("wave.png", wx.BITMAP_TYPE_ANY))

        # put some text
        txt_welcome = wx.StaticText(pnl, label="Welcome to", style=wx.ALIGN_CENTER)
        txt_appname = wx.StaticText(pnl, label="APPMAR 1.0")
        txt_appname.SetFont(wx.Font(20, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        
        # put some buttons
        btn_start = wx.Button(pnl, label="Start")
        btn_about = wx.Button(pnl, label="About APPMAR")

        # associate a handler function to the buttons
        btn_about.Bind(wx.EVT_BUTTON, self.on_about)
        btn_start.Bind(wx.EVT_BUTTON, self.on_start)

        # create a sizer to manage the layout of child widgets
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizerh = wx.BoxSizer(wx.HORIZONTAL)
        sizer_flags = wx.SizerFlags().Border().Center()
        sizer.Add(logo, sizer_flags)
        sizer.Add(txt_welcome, sizer_flags)
        sizer.Add(txt_appname, sizer_flags)
        sizer.Add(sizerh, sizer_flags)
        sizerh.Add(btn_start, sizer_flags)
        sizerh.AddSpacer(100)
        sizerh.Add(btn_about, sizer_flags)
        pnl.SetSizer(sizer)
        sizer.Fit(self)

    def on_start(self, event):
        """Hides the start frame and opens the main frame."""
        self.Close(True)
        frm_main = FrameMain(None, title="APPMAR 1.0")
        frm_main.Show()

    def on_about(self, event):
        """Hides the start frame and opens the about frame."""
        frm_about = FrameAbout(None, title="About APPMAR")
        frm_about.Show()

if __name__ == "__main__":
    APP = wx.App()
    FRM = FrameStart(None, title="APPMAR 1.0")
    FRM.Show()
    APP.MainLoop()
