from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
from neo.rawio import AxonRawIO

from ..core.acquisition import AcquisitionBandwidth
from ..core.calibration import CalibrationSet
from ..core.document import Document, TraceDocument
from ..core.types import (
    LOAD_CHUNK_SAMPLES,
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
    SampleRange,
    detect_dead,
    kind_for_units,
    normalise_units,
)
from .base import (
    BaseReader,
    Claim,
    ProgressFn,
    CancelFn,
    RecordingProbe,
    filesystem_refusal,
    foreign_format_redirect,
    read_magic,
    register,
)
from .contract import ReaderReport, clamp_request, window_loaded, window_t0
from .errors import CorruptFileError, LoadCancelled, ReaderError, UnsupportedFormatError

__all__ = ["AbfReader"]

# Number of strided samples used to derive the nominal command voltage from a VOLTAGE channel.
_VOLTAGE_PROBE_SAMPLES: int = 20_000

# ABF v1 float-format magic. CPYA is ALSO an ABF variant (claimed, so we can tell the truth about
# it in probe()); b"ABF2" is the v2 header. All three are OURS.
_ABF_MAGICS: frozenset[bytes] = frozenset({b"ABF ", b"ABF2", b"CPYA"})


@register
class AbfReader(BaseReader):
    """Axon Binary Format reader via neo.rawio.AxonRawIO.

    Deliberately NOT pyabf: pyabf 2.3.8 raises ValueError("Support for float data is not
    implemented") on these ABF v1 float-format files. neo.rawio decodes them.
    """

    format_name: ClassVar[str] = "Axon ABF"
    extensions: ClassVar[tuple[str, ...]] = (".abf",)

    # -- sniff -----------------------------------------------------------------------------------
    @classmethod
    def sniff(cls, path: Path) -> Claim:
        """ONLY "is this file mine". Decoding failures are raised from probe()/load().
        Not .abf -> NO. b"ABF "/b"ABF2"/b"CPYA" -> STRONG (CPYA is ours: it lets probe() say
        something true instead of the app reporting "no reader for .abf"). Else WEAK on extension.
        """
        if path.suffix.lower() != ".abf":
            return Claim.NO
        if read_magic(path, 4) in _ABF_MAGICS:
            return Claim.STRONG
        return Claim.WEAK

    # -- probe -----------------------------------------------------------------------------------
    def probe(self) -> RecordingProbe:
        """Cheap: parse_header() + get_signal_size(), no bulk reads. Does NOT set is_dead or the
        measured nominal voltage (both need sampled data; load() sets them)."""
        magic = read_magic(self.path, 4)
        if magic == b"CPYA":
            raise UnsupportedFormatError(
                f"{self.path.name} is a 'CPYA'-variant ABF. This build reads the standard "
                f"b'ABF ' float format only. Re-export from Clampex to open it here.")

        reader = self._open_reader()
        n = int(reader.get_signal_size(block_index=0, seg_index=0))
        self._refuse_zero_sample_header_over_data(reader, n)
        rate = self._rate(reader)
        t0 = float(reader.get_signal_t_start(0, 0))
        channels = self._channel_infos(reader)
        report = self._sweep_report(reader, n)

        meta = RecordingMeta(
            path=self.path,
            format_name=self.format_name,
            n_samples=n,
            sample_rate_hz=rate,
            rate_provenance=Provenance.FILE_META,
            channels=channels,
            t0_s=t0,
            notes=report.notes,
            issues=report.issues,
        )
        return RecordingProbe(meta=meta)

    # -- load ------------------------------------------------------------------------------------
    def load(self, sample_range: SampleRange | None = None,
             progress: ProgressFn | None = None,
             cancel: CancelFn | None = None) -> Document:
        """Eager, full-signal (or the chosen range). Reads in LOAD_CHUNK_SAMPLES chunks, polling
        progress/cancel between chunks, into preallocated contiguous 1-D float32 per channel.
        Runs detect_dead() per channel and derives the measured nominal voltage."""
        magic = read_magic(self.path, 4)
        if magic == b"CPYA":
            raise UnsupportedFormatError(
                f"{self.path.name} is a 'CPYA'-variant ABF. This build reads the standard "
                f"b'ABF ' float format only. Re-export from Clampex to open it here.")

        reader = self._open_reader()
        n_full = int(reader.get_signal_size(block_index=0, seg_index=0))
        self._refuse_zero_sample_header_over_data(reader, n_full)
        rate = self._rate(reader)
        t0 = float(reader.get_signal_t_start(0, 0))
        base_channels = self._channel_infos(reader)
        n_ch = len(base_channels)

        # Built BEFORE the clamp so the clamp can report a window that outran the file.
        report = self._sweep_report(reader, n_full)

        if sample_range is None:
            start, stop = 0, n_full
        else:
            start, stop = clamp_request(
                sample_range, n_full, name=self.path.name, report=report)

        n = stop - start

        # Preallocate one contiguous float32 array per channel. Writing chunk columns into these
        # yields C-contiguous 1-D arrays by construction — no ascontiguousarray needed.
        #
        # Deliberately NOT wired through io.base.DowncastAudit, unlike the HDF5/TDMS/text
        # readers: an ABF stores int16 counts or float32 samples (this build reads the v1
        # float format), so no value narrower than float32 can exist on disk to be flushed to
        # zero, and neo's rescale multiplies by gains of order 0.1-1, which cannot push a
        # representable float32 below the subnormal floor. If a future neo grows a float64
        # ABF variant, route its blocks through the shared audit the way tdms._read_into does.
        buffers: list[np.ndarray] = [np.empty(n, dtype=np.float32) for _ in range(n_ch)]

        if progress is not None:
            progress(0.0, f"Loading {self.path.name}")

        try:
            a = start
            while a < stop:
                if cancel is not None and cancel():
                    raise LoadCancelled(f"Loading {self.path.name} was cancelled.")
                b = min(a + LOAD_CHUNK_SAMPLES, stop)
                raw = reader.get_analogsignal_chunk(
                    block_index=0, seg_index=0, i_start=a, i_stop=b)
                flt = reader.rescale_signal_raw_to_float(raw, dtype="float32")
                off = a - start
                for ch in range(n_ch):
                    buffers[ch][off:off + (b - a)] = flt[:, ch]
                if progress is not None:
                    progress((b - start) / n, f"Loading {self.path.name}")
                a = b
        except LoadCancelled:
            raise
        except UnsupportedFormatError:
            raise
        except Exception as exc:  # neo/numpy failure mid-read -> the same classified refusal
            raise self._refusal(exc) from exc

        # Per-channel dead detection now that we have data.
        channels: list[ChannelInfo] = []
        for info, arr in zip(base_channels, buffers):
            is_dead, reason = detect_dead(arr)
            channels.append(replace(info, is_dead=is_dead, dead_reason=reason))
        channels_t = tuple(channels)

        nominal_mV, v_prov = self._measure_nominal_voltage(channels_t, buffers)

        meta = RecordingMeta(
            path=self.path,
            format_name=self.format_name,
            n_samples=n,
            sample_rate_hz=rate,
            rate_provenance=Provenance.FILE_META,
            channels=channels_t,
            # The window's own clock, not the file's — see contract.window_t0.
            t0_s=window_t0(t0, start, rate),
            loaded_range=window_loaded(sample_range, start, n),
            nominal_voltage_mV=nominal_mV,
            voltage_provenance=v_prov,
            notes=report.notes,
            issues=report.issues,
        )

        if progress is not None:
            progress(1.0, f"Loaded {self.path.name}")

        return TraceDocument(meta=meta, channels=tuple(buffers))

    # -- helpers ---------------------------------------------------------------------------------
    # ABF section pointers count 512-byte blocks from the start of the file.
    _ABF_BLOCK_BYTES: ClassVar[int] = 512

    def _refuse_zero_sample_header_over_data(self, reader: AxonRawIO, n: int) -> None:
        """A header that says "zero samples" over a data section full of them is DAMAGE, not
        an empty recording — refuse it here, before the empty-recording door can.

        That door's sentence ends "this file is the expected result and can be deleted",
        which is true of an acquisition that was armed and stopped without recording (a bare
        header, nothing after it) and catastrophically false of an acquisition that CRASHED:
        the header's sample counts were initialised to zero before the data was flushed, so
        the counts lie while the data section holds the whole recording. [VERIFIED] on the
        owner's 9,001,992-byte quick ABF with lActualAcqLength/lSynchArraySize zeroed:
        ``get_signal_size()`` returns 0 while 8,995,848 bytes of float samples sit past the
        header, all of them recoverable by restoring the two header ints. The TDMS reader
        already routes its analogous case around that door (`_scan`: "a cut recording is
        neither expected nor safe to delete"); this is the same reroute for ABF.

        A genuinely tiny file — header only, no data section bytes — falls through to the
        empty-recording sentence, which is the honest one for it. The data-section offset is
        taken from the header neo just parsed; if a neo version stops exposing it, this check
        silently stands down and the empty door answers as before (an unverifiable suspicion
        must not turn real empty files into "damaged" ones).
        """
        if n > 0:
            return
        try:
            data_ptr = int(reader._axon_info["lDataSectionPtr"])
        except Exception:
            return
        if data_ptr <= 0:
            return
        data_start = data_ptr * self._ABF_BLOCK_BYTES
        data_bytes = self.path.stat().st_size - data_start
        if data_bytes <= 0:
            return
        raise CorruptFileError(
            f"{self.path.name}: the header claims this recording holds no samples, but the "
            f"file carries {data_bytes:,} bytes of sample data past its header — what an "
            f"acquisition that crashed before finalising the header leaves behind, not an "
            f"empty recording. The samples are still in the file, so do not delete it: "
            f"recover the original from the acquisition computer, or re-export it from "
            f"Clampex, to open the recording here.")

    def _rate(self, reader: AxonRawIO) -> float:
        """The header's declared rate, refused when it is not one.

        The rate is the denominator of every dwell time and the divisor in ``duration_s``. A
        zero or non-finite one does not produce an error anywhere downstream — it produces
        ``inf`` seconds and a plot with no time axis — so it is refused here, where the file can
        still be named, rather than surfacing as a division much later with nothing attached.
        """
        rate = float(reader.get_signal_sampling_rate())
        if not np.isfinite(rate) or rate <= 0.0:
            raise CorruptFileError(
                f"{self.path.name}: the header declares a sample rate of {rate}, which is not a "
                f"rate. Every dwell time and duration in this app divides by it.")
        return rate

    @staticmethod
    def _sweep_report(reader: AxonRawIO, n: int) -> ReaderReport:
        """Say so when this ABF holds more than the one sweep that gets loaded.

        ``get_signal_size(block_index=0, seg_index=0)`` is the length of the FIRST sweep, and
        every number this reader reports — ``n_samples``, the duration, the denominator of every
        event rate — is that sweep's. A gap-free recording has exactly one, which is what every
        file in the owner's corpus is ([VERIFIED] all six ABFs: 1 block, 1 segment). An episodic
        protocol has many, and on one of those the app would describe a fraction of the file
        without anything saying that is what it was doing.

        Reported rather than refused: the first sweep IS a real recording and is analysable.
        What is not acceptable is loading it under the whole file's name in silence.
        """
        report = ReaderReport()
        try:
            n_blocks = int(reader.block_count())
            n_sweeps = int(reader.segment_count(0))
        except Exception:      # a neo version that does not expose the counts: say nothing
            return report
        if n_blocks <= 1 and n_sweeps <= 1:
            return report
        report.issue(
            f"this ABF holds {n_sweeps:,} sweeps in {n_blocks:,} block(s) — an episodic protocol, "
            f"not one gap-free recording. Only the first sweep ({n:,} samples) is loaded, so the "
            f"duration, event count and every rate derived from them describe that sweep alone. "
            f"Re-export it from Clampex as a single gap-free trace to analyse the whole run.")
        return report

    def _open_reader(self) -> AxonRawIO:
        """parse_header(), converting any neo failure into the user-facing sentence
        io/errors.py promises. The CPYA case is intercepted by the caller before we get here."""
        try:
            reader = AxonRawIO(filename=str(self.path))
            reader.parse_header()
        except Exception as exc:
            raise self._refusal(exc) from exc
        return reader

    def _refusal(self, exc: Exception) -> ReaderError:
        """The user-facing refusal for a neo failure, classified from the FILE, not from ``exc``.

        neo's own words for these failures — "'NoneType' object is not subscriptable" for an
        empty or misnamed file, "[Errno 2] No such file or directory: '/abs/path'" for a gone
        one, "mmap length is greater than file size" for a cut one — identify nothing a user
        can act on, and io/errors.py promises they never reach the UI. The exception is only
        whatever neo happened to trip over first, so the real cause is re-derived from the
        bytes on disk, worded the way the other three readers word the identical inputs — which
        is now literally the same words rather than a resemblance: both classifications below
        are ``io.base``'s, and all four readers call them."""
        gone_locked_or_folder = filesystem_refusal(self.path, exc)
        if gone_locked_or_folder is not None:
            # Order matters and belongs to the shared classifier: a mode-000 FOLDER used to be
            # called a permission problem here (PermissionError was tested first) and a folder
            # by the .h5/.tdms readers, so one input had two dialogs and this one sent the user
            # to fix a permission that was never the reason.
            return CorruptFileError(gone_locked_or_folder)
        renamed = foreign_format_redirect(self.path)
        if renamed is not None:
            # A file of some other format wearing an .abf name. Was: only HDF5 bytes were
            # identified here; a TDMS or a text export named .abf fell through to the generic
            # "does not start with an ABF signature" below, which is true and useless. The
            # shared redirect names whatever the bytes really are, for every format the app
            # reads, in every direction.
            return UnsupportedFormatError(renamed)
        magic = read_magic(self.path, 8)
        if self.path.stat().st_size == 0:
            return CorruptFileError(
                f"{self.path.name} is empty — 0 bytes, not even an ABF header. The usual "
                f"cause is an interrupted copy or an acquisition that never wrote data; "
                f"recover the original file.")
        if magic[:4] not in _ABF_MAGICS:
            return CorruptFileError(
                f"{self.path.name} is named .abf but does not start with an ABF signature, "
                f"so it is not an ABF file. If it was renamed, restore its original "
                f"extension; otherwise re-export the recording from Clampex.")
        return CorruptFileError(
            f"{self.path.name} cannot be decoded as an ABF: the file is shorter than its "
            f"header promises, or the header itself is damaged — the usual cause is an "
            f"interrupted copy or download. Recover the original file.")

    @staticmethod
    def _channel_infos(reader: AxonRawIO) -> tuple[ChannelInfo, ...]:
        """Build ChannelInfo from header['signal_channels']. Units are spelling-normalised only
        ('PA'->'pA', 'V' stays 'V'); kind is derived from the normalised units. is_dead is left
        False here — it needs sampled data and is set in load()."""
        sc = reader.header["signal_channels"]
        infos: list[ChannelInfo] = []
        for i, row in enumerate(sc):
            rec = dict(zip(sc.dtype.names, row))
            raw_units = normalise_units(str(rec["units"]))
            infos.append(ChannelInfo(
                index=i,
                name=str(rec["name"]),
                raw_units=raw_units,
                kind=kind_for_units(raw_units),
                acquisition_bandwidth=AbfReader._acquisition_bandwidth(reader, rec.get("id")),
            ))
        return tuple(infos)

    @staticmethod
    def _acquisition_bandwidth(reader: AxonRawIO, signal_id) -> AcquisitionBandwidth:
        """Read channel-aligned declarations already parsed by Neo; never read samples.

        Mapping follows Neo's AxonRawIO._parse_header: ABF1 signal ``id`` is the
        physical ADC index from nADCSamplingSeq, whereas ABF2 ``id`` indexes
        listADCInfo (its nADCNum is NOT the list index). Current Neo exposes no
        ABF1 low-pass fields, so those files honestly remain unknown. Optional
        per-ADC arrays are supported if a future Neo exposes the same ABF fields;
        no unverified binary offsets are introduced here.

        Field/sentinel definitions are documented by the pyABF project in its
        SDK-derived ABF1 reference:
        https://github.com/swharden/pyABF/blob/main/website/content/abf1-file-format/index.md
        Signal low-pass 100000 means bypass, not a 100 kHz corner. The telegraph
        100000 default is kept unknown rather than used as a measured bandwidth.
        """
        header = getattr(reader, "_axon_info", None)
        if not isinstance(header, dict):
            return AcquisitionBandwidth(note="Acquisition cutoff is not exposed by the ABF reader.")
        try:
            version = float(header["fFileVersionNumber"])
            channel_id = int(signal_id)
            if not np.isfinite(version) or version < 1 or channel_id < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            return AcquisitionBandwidth(source="file_header", note=(
                "Acquisition cutoff is unknown: ABF channel/header mapping is unavailable."))

        names = ("nTelegraphEnable", "nTelegraphInstrument", "fTelegraphFilter",
                 "fSignalLowpassFilter", "nLowpassFilterType",
                 "fPostProcessLowpassFilter", "nPostProcessLowpassFilterType")
        values: dict = {}
        raw: list[tuple[str, str]] = [("neo_signal_id", str(signal_id))]
        if version < 2:
            for name in names:
                try:
                    values[name] = header[name][channel_id]
                except (KeyError, IndexError, TypeError):
                    pass
            raw.append(("physical_ADC_index", str(channel_id)))
        else:
            try:
                adc = header["listADCInfo"][channel_id]
                if not isinstance(adc, dict):
                    raise TypeError
            except (KeyError, IndexError, TypeError):
                return AcquisitionBandwidth(source="file_header", header_fields=tuple(raw), note=(
                    "Acquisition cutoff is unknown: the ABF2 ADC entry could not be mapped."))
            values = {name: adc[name] for name in names if name in adc}
            if "nADCNum" in adc:
                raw.append(("nADCNum", str(adc["nADCNum"])))
        raw.extend((name, str(value)) for name, value in values.items())

        def number(name: str) -> float | None:
            value = values.get(name)
            if isinstance(value, (bool, np.bool_)):
                return None
            try:
                result = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return result if np.isfinite(result) else None

        notes = ["File-header declaration only; verify against the acquisition protocol."]
        candidates: list[tuple[float, str, str]] = []
        telegraph = number("fTelegraphFilter")
        telegraph_enabled = number("nTelegraphEnable") == 1
        if (telegraph_enabled and telegraph is not None
                and telegraph > 0 and telegraph != 100000):
            candidates.append((telegraph, "", "amplifier telegraph"))
        elif "fTelegraphFilter" in values:
            notes.append("Amplifier cutoff is unknown: telegraph is disabled, invalid, or uses its default value.")

        signal = number("fSignalLowpassFilter")
        filter_code = number("nLowpassFilterType")
        filter_name = {1: "External", 2: "RC", 3: "Bessel", 4: "Butterworth"}.get(filter_code, "")
        signal_bypassed = signal == 100000
        if signal is not None and signal > 0 and not signal_bypassed:
            if filter_code == 0:
                notes.append("Signal-conditioner cutoff conflicts with the header's no-filter type; not used.")
            else:
                candidates.append((signal, filter_name, "signal conditioner"))
        elif signal_bypassed:
            notes.append("The signal-conditioner stage is reported bypassed; this does not establish an unfiltered acquisition chain.")
        elif "fSignalLowpassFilter" in values:
            notes.append("Signal-conditioner cutoff is absent or invalid; no bandwidth is inferred.")

        if "fPostProcessLowpassFilter" in values:
            notes.append("ABF post-processing low-pass is preserved separately in header fields, not treated as hardware bandwidth.")
        if candidates:
            cutoff, filter_type, stage = min(candidates, key=lambda item: item[0])
            notes.append(f"Reported cutoff is the narrowest declared hardware stage ({stage}), not the combined effective response.")
            notes.append("Filter order and full transfer function are unknown; no rise time is inferred.")
            return AcquisitionBandwidth(cutoff, "reported", "file_header", filter_type,
                                        note=" ".join(notes), header_fields=tuple(raw))
        if version < 2 and not any(name in values for name in ("fTelegraphFilter", "fSignalLowpassFilter")):
            notes.append("The current Neo ABF1 parser does not expose acquisition cutoff fields.")
        else:
            notes.append("No usable hardware cutoff is available; unknown does not mean unfiltered.")
        return AcquisitionBandwidth(status="disabled" if signal_bypassed else "unknown",
                                    source="file_header", note=" ".join(notes), header_fields=tuple(raw))

    @staticmethod
    def _measure_nominal_voltage(channels: tuple[ChannelInfo, ...],
                                 buffers: list[np.ndarray]) -> tuple[float | None, Provenance]:
        """MEASURED nominal command voltage: median of a strided sample of the VOLTAGE channel,
        scaled by the default voltage gain (0.1; ABF stores it 10x too large). Returns
        (None, UNKNOWN) when there is no VOLTAGE channel.

        [VERIFIED] IN14 median -6998.90 mV x 0.1 = -699.89 mV -> voltage_group_mV -700."""
        for info in channels:
            if info.kind is ChannelKind.VOLTAGE:
                arr = buffers[info.index]
                step = max(1, len(arr) // _VOLTAGE_PROBE_SAMPLES)
                sample = np.asarray(arr[::step], dtype=np.float64)
                med = float(np.median(sample))
                # Same detection the Session seeds with, from the same number -- so the
                # command-voltage readout and the calibrated channel can never disagree
                # about whether this file uses the 10x-inflated encoding.
                gain = CalibrationSet.default_for(info, med).gain
                return med * gain, Provenance.MEASURED
        return None, Provenance.UNKNOWN
