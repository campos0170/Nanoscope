"""Nonlinear X-Y regression, deliberately separate from event-density likelihoods.

All models minimize residuals in the original y units. Inputs, options, fitted
curves and diagnostics travel together in a versioned, JSON-safe result. This
module has no Qt dependency and can run in a cancellable worker.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, fields
import math
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from scipy.special import erf, expit, voigt_profile
from scipy.stats import t as student_t, norm


class FitCancelled(RuntimeError):
    """The caller cancelled a fit; no partial result should replace a saved fit."""


def _safe_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(v) for v in value]
    return value


@dataclass
class CurveFitResult:
    parameters: dict[str, float]
    standard_errors: dict[str, float | None]
    parameter_ci: dict[str, tuple[float, float] | None]
    x: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray
    residuals: np.ndarray
    x_curve: np.ndarray
    y_curve: np.ndarray
    confidence_lower: np.ndarray | None
    confidence_upper: np.ndarray | None
    prediction_lower: np.ndarray | None
    prediction_upper: np.ndarray | None
    metrics: dict[str, Any]
    warnings: list[str]
    config: dict[str, Any]
    success: bool = True
    message: str = ''
    y_sigma: np.ndarray | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return _safe_json({f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'CurveFitResult':
        """Validate a saved result without fitting or evaluating its equation.

        Missing optional diagnostics are valid older/minimal results. Malformed
        diagnostics are rejected, never partially aligned to a different sample.
        Limits protect project loading from accidentally enormous saved grids.
        """
        if not isinstance(data, Mapping):
            raise ValueError('Saved nonlinear result must be a mapping.')
        version = data.get('schema_version', 1)
        if isinstance(version, bool) or version != 1:
            raise ValueError('Unsupported nonlinear fit result version.')
        config = data.get('config')
        if not isinstance(config, Mapping):
            raise ValueError('Saved nonlinear result lacks its model configuration.')
        config = dict(config)
        model_id = config.get('model', 'gaussian')
        if not isinstance(model_id, str):
            raise ValueError('Saved model identifier must be a string.')
        model = get_model(model_id)
        if model.id == 'custom':
            names = config.get('custom_parameters', ())
            expression = config.get('custom_expression', '')
            if not isinstance(names, (list, tuple)) or any(not isinstance(p, str) for p in names) or not isinstance(expression, str):
                raise ValueError('Saved custom equation and parameter names are malformed.')
            # Parsing/validation only: never evaluate a saved expression on open.
            compile_custom_expression(expression, names)
            names = tuple(names)
        else:
            names = model.parameter_names
        raw_parameters = data.get('parameters')
        if not isinstance(raw_parameters, Mapping) or set(raw_parameters) != set(names):
            raise ValueError('Saved parameter names do not match the declared model.')

        def number(value, label):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f'Saved {label} must be a finite number.')
            try:
                out = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Saved {label} must be a finite number.') from exc
            if not np.isfinite(out):
                raise ValueError(f'Saved {label} must be a finite number.')
            return out

        def typed_number(value, label):
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f'Saved {label} must be a numeric value, not text or a container.')
            return number(value, label)

        for key, allowed in (('weighting', ('uniform', 'sigma')),
                             ('loss', ('linear', 'soft_l1', 'huber'))):
            if key in config and config[key] not in allowed:
                raise ValueError(f'Saved {key} configuration is unsupported.')
        if 'confidence' in config and not .5 <= typed_number(config['confidence'], 'confidence') < 1:
            raise ValueError('Saved confidence must be at least 0.5 and below 1.')
        for key in ('x_min', 'x_max', 'f_scale'):
            if config.get(key) is not None:
                value = typed_number(config[key], key)
                if key == 'f_scale' and value <= 0:
                    raise ValueError('Saved robust residual scale must be positive.')
        if config.get('x_min') is not None and config.get('x_max') is not None and config['x_min'] >= config['x_max']:
            raise ValueError('Saved x limits are reversed or empty.')
        if 'max_nfev' in config:
            value = typed_number(config['max_nfev'], 'maximum evaluations')
            if value < 1 or not value.is_integer():
                raise ValueError('Saved maximum evaluations must be a positive integer.')
        for key in ('initial', 'fixed', 'bounds'):
            group = config.get(key)
            if group is None:
                config[key] = {}
                continue
            if not isinstance(group, Mapping) or set(group)-set(names):
                raise ValueError(f'Saved {key} configuration has invalid parameter names.')
            for name, value in group.items():
                if key == 'bounds':
                    if not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (2,):
                        raise ValueError(f'Saved bounds for {name} must have two endpoints.')
                    low = -np.inf if value[0] is None else typed_number(value[0], f'lower bound for {name}')
                    high = np.inf if value[1] is None else typed_number(value[1], f'upper bound for {name}')
                    if low >= high:
                        raise ValueError(f'Saved bounds for {name} are reversed or empty.')
                else:
                    typed_number(value, f'{key} value for {name}')

        params = {p: number(raw_parameters[p], f'parameter {p}') for p in names}
        values = dict(parameters=params, config=config, schema_version=1)

        def array(key, *, expected=None, optional=False, maximum=5_000_000):
            value = data.get(key)
            if optional and value is None:
                return None
            if value is None:
                raise ValueError(f'Saved nonlinear result is missing {key}.')
            try:
                if len(value) > maximum:
                    raise ValueError(f'Saved {key} exceeds the supported point limit ({maximum}).')
                result = np.asarray(value, dtype=float)
            except (TypeError, OverflowError) as exc:
                raise ValueError(f'Saved {key} must be a one-dimensional numeric array.') from exc
            if result.ndim != 1 or result.size < 2 or result.size > maximum:
                raise ValueError(f'Saved {key} must be a bounded one-dimensional array with at least two points.')
            if expected is not None and result.size != expected:
                raise ValueError(f'Saved {key} length does not match the corresponding coordinates.')
            if not np.all(np.isfinite(result)):
                raise ValueError(f'Saved {key} contains non-finite values.')
            return result

        values['x'] = array('x')
        n = values['x'].size
        for key in ('y', 'y_fit', 'residuals'):
            values[key] = array(key, expected=n)
        values['x_curve'] = array('x_curve', maximum=100_000)
        m = values['x_curve'].size
        values['y_curve'] = array('y_curve', expected=m, maximum=100_000)
        for prefix in ('confidence', 'prediction'):
            lower = array(prefix+'_lower', expected=m, optional=True, maximum=100_000)
            upper = array(prefix+'_upper', expected=m, optional=True, maximum=100_000)
            if (lower is None) != (upper is None):
                raise ValueError(f'Saved {prefix} bands must contain both lower and upper arrays.')
            if lower is not None and np.any(lower > upper):
                raise ValueError(f'Saved {prefix} lower band exceeds its upper band.')
            values[prefix+'_lower'], values[prefix+'_upper'] = lower, upper
        values['y_sigma'] = array('y_sigma', expected=n, optional=True)
        if values['y_sigma'] is not None and np.any(values['y_sigma'] <= 0):
            raise ValueError('Saved observation uncertainties must be positive.')
        if config.get('weighting', 'uniform') == 'sigma' and values['y_sigma'] is None:
            raise ValueError('Saved sigma-weighted result is missing observation uncertainties.')
        # A result must preserve the same residual convention as its graph/export.
        expected_residuals = values['y']-values['y_fit']
        tolerance = max(float(np.max(np.abs(values['y']))),
                        float(np.max(np.abs(values['y_fit']))), 1e-100)*1e-12
        if not np.allclose(values['residuals'], expected_residuals, rtol=1e-9, atol=tolerance):
            raise ValueError('Saved residuals do not match y minus the fitted response.')

        for key in ('standard_errors', 'parameter_ci'):
            raw = data.get(key, {})
            if not isinstance(raw, Mapping) or set(raw)-set(names):
                raise ValueError(f'Saved {key} names do not match the declared model.')
            parsed = {}
            for name in names:
                value = raw.get(name)
                if value is None:
                    parsed[name] = None
                elif key == 'standard_errors':
                    value = number(value, f'standard error for {name}')
                    if value < 0: raise ValueError('Saved standard errors cannot be negative.')
                    parsed[name] = value
                else:
                    if not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (2,):
                        raise ValueError(f'Saved parameter interval for {name} must have two bounds.')
                    low, high = (number(v, f'interval for {name}') for v in value)
                    if low > high: raise ValueError(f'Saved parameter interval for {name} is reversed.')
                    parsed[name] = (low, high)
            values[key] = parsed
        metrics = data.get('metrics', {})
        if not isinstance(metrics, Mapping):
            raise ValueError('Saved fit metrics must be a mapping.')
        count_metrics = {'n', 'n_input', 'n_excluded_range', 'n_parameters', 'dof', 'jacobian_rank', 'nfev'}
        numeric_metrics = {'sse', 'rmse', 'r_squared', 'r2', 'adjusted_r_squared', 'chi_squared',
                           'reduced_chi_squared', 'aic', 'bic', 'jacobian_condition'}
        text_metrics = {'objective', 'prediction_noise', 'interval_method'}
        for key, value in metrics.items():
            if not isinstance(key, str) or not isinstance(value, (str, bool, int, float, np.number, type(None))):
                raise ValueError('Saved fit metrics must contain scalar values.')
            if key in count_metrics:
                count = typed_number(value, f'metric {key}')
                if count < 0 or not count.is_integer():
                    raise ValueError(f'Saved metric {key} must be a non-negative integer.')
            elif key in numeric_metrics and value is not None:
                typed_number(value, f'metric {key}')
            elif key in text_metrics and not isinstance(value, str):
                raise ValueError(f'Saved metric {key} must be text.')
            elif isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                number(value, f'metric {key}')
        if 'n' in metrics and metrics['n'] != n:
            raise ValueError('Saved fit observation count does not match its arrays.')
        if 'n_input' in metrics and metrics['n_input'] < n:
            raise ValueError('Saved input count is smaller than the fitted observation count.')
        if 'n_excluded_range' in metrics and 'n_input' in metrics and metrics['n_excluded_range'] != metrics['n_input']-n:
            raise ValueError('Saved excluded observation count is inconsistent.')
        if 'n_parameters' in metrics:
            k = metrics['n_parameters']
            if k > len(names) or k >= n or ('dof' in metrics and metrics['dof'] != n-k):
                raise ValueError('Saved parameter count or degrees of freedom is inconsistent.')
            if 'jacobian_rank' in metrics and metrics['jacobian_rank'] > k:
                raise ValueError('Saved Jacobian rank exceeds the free parameter count.')
        values['metrics'] = dict(metrics)
        warning_list = data.get('warnings', [])
        if not isinstance(warning_list, (list, tuple)) or any(not isinstance(w, str) for w in warning_list):
            raise ValueError('Saved warnings must be a list of strings.')
        values['warnings'] = list(warning_list)
        values['success'] = data.get('success', True)
        values['message'] = data.get('message', '')
        if not isinstance(values['success'], (bool, np.bool_)) or not isinstance(values['message'], str):
            raise ValueError('Saved optimizer status is malformed.')
        return cls(**values)


@dataclass(frozen=True)
class CurveModel:
    id: str
    label: str
    parameter_names: tuple[str, ...]
    formula: str
    description: str
    positive_x: bool = False
    strictly_positive_x: bool = False

    def guess(self, x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
        xx, yy = _xy(x, y)
        return _guess(self.id, xx, yy)

    def default_bounds(self, x: Sequence[float], y: Sequence[float]) -> dict[str, tuple[float, float]]:
        xx, yy = _xy(x, y)
        span = max(float(np.ptp(xx)), abs(float(np.median(xx))) * 1e-10, 1e-12)
        tiny = max(span * 1e-10, np.finfo(float).tiny ** .25)
        out = {p: (-np.inf, np.inf) for p in self.parameter_names}
        for p in ('sigma', 'gamma', 'tau', 'ec50', 'midpoint_scale') + (('slope',) if self.id == 'boltzmann' else ()):
            if p in out:
                out[p] = (tiny, np.inf)
        for p in ('hill', 'beta', 'asymmetry'):
            if p in out:
                out[p] = (1e-4, 100.0)
        return out

    def evaluate(self, x: Sequence[float], parameters: Mapping[str, float]) -> np.ndarray:
        return evaluate_curve(self.id, x, parameters)


_MODELS = (
    CurveModel('linear', 'Straight line', ('intercept', 'slope'), 'intercept + slope*x',
               'Linear regression in original y units with optional weights, bounds and robust loss.'),
    CurveModel('gaussian', 'Gaussian peak + baseline', ('baseline', 'amplitude', 'center', 'sigma'),
               'baseline + amplitude * exp(-0.5*((x-center)/sigma)**2)',
               'Gaussian X-Y peak. Amplitude is height; sigma is standard deviation in x units, FWHM=2.35482*sigma. Distinct from population-density fitting.'),
    CurveModel('exp1', 'Single exponential + baseline', ('baseline', 'amplitude', 'rate'),
               'baseline + amplitude * exp(rate*x)',
               'Original-y-space regression. Signed rate has inverse-x units; a negative rate describes decay, with tau=-1/rate.'),
    CurveModel('exp2', 'Double exponential + baseline', ('baseline', 'amplitude1', 'rate1', 'amplitude2', 'rate2'),
               'baseline + amplitude1*exp(rate1*x) + amplitude2*exp(rate2*x)',
               'Two exponential components. Similar rates may be unidentifiable; component labels are interchangeable.'),
    CurveModel('power_law', 'Power law + baseline', ('baseline', 'amplitude', 'exponent'),
               'baseline + amplitude * x**exponent',
               'Power law fitted in original y units, requiring strictly positive x. Units of amplitude depend on exponent.', True, True),
    CurveModel('logistic4', 'Logistic / dose response (4PL)', ('bottom', 'top', 'ec50', 'hill'),
               'bottom + (top-bottom) / (1 + (ec50/x)**hill)',
               'Positive-dose sigmoid. ec50 is the half-response concentration; hill is positive and top may be below bottom. Requires x >= 0.', True),
    CurveModel('logistic5', 'Asymmetric logistic (5PL)', ('bottom', 'top', 'midpoint_scale', 'hill', 'asymmetry'),
               'bottom + (top-bottom) / (1 + (midpoint_scale/x)**hill)**asymmetry',
               'Positive-dose asymmetric sigmoid. midpoint_scale is NOT EC50 when asymmetry differs from one. Requires x >= 0.', True),
    CurveModel('hill', 'Hill saturation', ('bottom', 'top', 'ec50', 'hill'),
               'bottom + (top-bottom)*x**hill/(ec50**hill+x**hill)',
               'Hill saturation with baseline and asymptote. ec50 is half-response; x is non-negative concentration.', True),
    CurveModel('boltzmann', 'Boltzmann / voltage sigmoid', ('bottom', 'top', 'v_half', 'slope'),
               'bottom + (top-bottom)/(1+exp(-(x-v_half)/slope))',
               'Sigmoid allowing signed x, suitable for voltage. v_half and positive slope are in x units; decreasing curves have top < bottom.'),
    CurveModel('stretched_exp', 'Stretched exponential + baseline', ('baseline', 'amplitude', 'tau', 'beta'),
               'baseline + amplitude*exp(-(x/tau)**beta)',
               'Non-negative time x and positive tau/beta. beta=1 is ordinary exponential decay.', True),
    CurveModel('lorentzian', 'Lorentzian peak + baseline', ('baseline', 'amplitude', 'center', 'gamma'),
               'baseline + amplitude/(1+((x-center)/gamma)**2)',
               'Amplitude is peak height; gamma is half width at half maximum in x units.'),
    CurveModel('voigt', 'Voigt peak + baseline', ('baseline', 'amplitude', 'center', 'sigma', 'gamma'),
               'baseline + amplitude*Voigt(x-center,sigma,gamma)/Voigt(0,sigma,gamma)',
               'Height-normalized convolution of Gaussian and Lorentzian peaks. sigma is Gaussian standard deviation; gamma is Lorentzian half width.'),
    CurveModel('custom', 'Custom equation', (), 'User equation in x and named parameters',
               'Safe arithmetic expression. Functions: exp, log, log10, sqrt, abs, sin, cos, tanh, erf, expit. No Python statements, attributes or file access.'),
)
_MODEL_MAP = {m.id: m for m in _MODELS}


def list_models() -> list[CurveModel]:
    return list(_MODELS)


def get_model(model_id: str) -> CurveModel:
    try:
        return _MODEL_MAP[model_id]
    except KeyError:
        raise ValueError(f'Unknown curve model: {model_id!r}.') from None


def _xy(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xx.ndim != 1 or yy.ndim != 1 or xx.size != yy.size or xx.size < 2:
        raise ValueError('x and y must be equal-length one-dimensional arrays with at least two observations.')
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
        raise ValueError('x and y must contain only finite observations; remove or resolve missing values explicitly.')
    if np.ptp(xx) == 0:
        raise ValueError('Curve fitting requires at least two distinct x values.')
    return xx, yy


def _guess(model_id: str, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    order = np.argsort(x); xx, yy = x[order], y[order]
    span = max(float(np.ptp(xx)), 1e-12)
    nedge = max(1, len(xx)//10)
    left, right = float(np.median(yy[:nedge])), float(np.median(yy[-nedge:]))
    baseline = .5*(left+right)
    smooth = median_filter(yy[:, None], size=(3, 1), mode='nearest')[:, 0] if len(yy) >= 7 else yy
    peak = int(np.argmax(np.abs(smooth-baseline)))
    amplitude = float(smooth[peak]-baseline)
    positive = np.maximum(np.sign(amplitude)*(smooth-baseline), 0)
    width = math.sqrt(float(np.sum(positive*(xx-xx[peak])**2)/np.sum(positive))) if np.sum(positive) else span/6
    width = min(max(width, span/100), span/2)
    center = float(xx[peak])
    if model_id == 'linear':
        slope = float(np.dot(xx-np.mean(xx), yy-np.mean(yy))/np.sum((xx-np.mean(xx))**2))
        return dict(intercept=float(np.mean(yy)-slope*np.mean(xx)), slope=slope)
    if model_id in ('gaussian', 'lorentzian', 'voigt'):
        d = dict(baseline=baseline, amplitude=amplitude or 1., center=center)
        if model_id in ('gaussian', 'voigt'): d['sigma'] = width
        if model_id in ('lorentzian', 'voigt'): d['gamma'] = width/2
        return d
    if model_id in ('exp1', 'exp2', 'stretched_exp'):
        baseline = right - .05*(left-right)
        rate = -2/span
        amplitude = (left-baseline)*float(np.exp(np.clip(-rate*xx[0], -200, 200)))
        if model_id in ('exp1', 'exp2'):
            # Three equally spaced samples eliminate the offset: successive
            # increments have ratio exp(rate*dx), including growth and signed x.
            gx = xx[0]+span*np.array([.1,.5,.9])
            gy = np.interp(gx,xx,smooth)
            d1,d2 = np.diff(gy)
            if d1*d2 > 0 and abs(d1) > 1e-12*max(float(np.ptp(yy)),1e-100):
                ratio=d2/d1
                if abs(ratio-1) > 1e-4:
                    rate=float(np.log(ratio)/(.4*span))
                    baseline=float((gy[1]-ratio*gy[0])/(1-ratio))
                    amplitude=float((gy[0]-baseline)*np.exp(np.clip(-rate*gx[0],-200,200)))
            if model_id == 'exp1': return dict(baseline=baseline, amplitude=amplitude or 1., rate=rate)
            return dict(baseline=baseline, amplitude1=.7*amplitude, rate1=rate*.5,
                        amplitude2=.3*amplitude, rate2=rate*2.5)
        return dict(baseline=baseline, amplitude=left-baseline, tau=span/2, beta=1.)
    if model_id == 'power_law':
        return dict(baseline=0., amplitude=float(np.median(yy))/max(float(np.median(xx)), 1e-12), exponent=1.)
    if model_id in ('logistic4', 'logistic5', 'hill'):
        positive_x = xx[xx > 0]
        midpoint = float(np.exp(np.mean(np.log(positive_x)))) if positive_x.size else 1.
        d = dict(bottom=left, top=right, hill=1.)
        d['midpoint_scale' if model_id == 'logistic5' else 'ec50'] = midpoint
        if model_id == 'logistic5': d['asymmetry'] = 1.
        return d
    if model_id == 'boltzmann':
        return dict(bottom=left, top=right, v_half=float(xx[np.argmin(abs(yy-(left+right)/2))]), slope=span/8)
    return {}


_FUNCTIONS: dict[str, Callable] = {
    'exp': np.exp, 'log': np.log, 'log10': np.log10, 'sqrt': np.sqrt,
    'abs': np.abs, 'sin': np.sin, 'cos': np.cos, 'tanh': np.tanh,
    'erf': erf, 'expit': expit,
}
_CONSTANTS = {'pi': np.pi, 'e': np.e}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.divide, ast.Pow: np.power}


def compile_custom_expression(expression: str, parameter_names: Sequence[str]) -> Callable:
    """Compile a restricted arithmetic AST by interpretation, never Python eval."""
    if not expression or len(expression) > 2000:
        raise ValueError('Custom equation must contain 1–2000 characters.')
    names = tuple(parameter_names)
    if not names or len(names) > 20 or len(set(names)) != len(names):
        raise ValueError('Provide 1–20 unique custom parameter names.')
    reserved = {'x', *_FUNCTIONS, *_CONSTANTS}
    if any(not isinstance(n, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', n) or n in reserved for n in names):
        raise ValueError('Parameter names must be identifiers distinct from x, constants and function names.')
    try:
        tree = ast.parse(expression, mode='eval').body
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('Invalid custom equation syntax.') from exc
    if len(list(ast.walk(tree))) > 150:
        raise ValueError('Custom equation is too complex.')
    used: set[str] = set()

    def validate(node: ast.AST, depth: int = 0) -> None:
        if depth > 25: raise ValueError('Custom equation is too deeply nested.')
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)) or abs(node.value) > 1e100 or not np.isfinite(float(node.value)):
                raise ValueError('Only finite numeric constants are allowed.')
        elif isinstance(node, ast.Name):
            if node.id not in {'x', *names, *_CONSTANTS}: raise ValueError(f'Unknown equation name: {node.id}.')
            used.add(node.id)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            validate(node.left, depth+1); validate(node.right, depth+1)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            validate(node.operand, depth+1)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and len(node.args) == 1 and not node.keywords:
            validate(node.args[0], depth+1)
        else:
            raise ValueError('Only arithmetic, x, parameters and the listed mathematical functions are allowed.')
    validate(tree)
    unused = set(names)-used
    if unused: raise ValueError('Unused custom parameters: '+', '.join(sorted(unused))+'.')
    if 'x' not in used: raise ValueError('Custom equation must use x.')

    def evaluate_node(node: ast.AST, env: dict) -> Any:
        if isinstance(node, ast.Constant): return float(node.value)
        if isinstance(node, ast.Name): return env[node.id]
        if isinstance(node, ast.BinOp): return _BINOPS[type(node.op)](evaluate_node(node.left, env), evaluate_node(node.right, env))
        if isinstance(node, ast.UnaryOp):
            value = evaluate_node(node.operand, env)
            return -value if isinstance(node.op, ast.USub) else value
        return _FUNCTIONS[node.func.id](evaluate_node(node.args[0], env))

    def evaluate(x: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
        with np.errstate(all='ignore'):
            return np.broadcast_to(np.asarray(evaluate_node(tree, {**_CONSTANTS, **p, 'x': x}), dtype=float), np.shape(x))
    return evaluate


def evaluate_curve(model_id: str, x: Sequence[float], parameters: Mapping[str, float],
                   *, custom_expression: str = '', custom_parameters: Sequence[str] = ()) -> np.ndarray:
    xx = np.asarray(x, dtype=float); p = parameters
    if model_id == 'custom': return compile_custom_expression(custom_expression, custom_parameters)(xx, p)
    get_model(model_id)
    with np.errstate(all='ignore'):
        if model_id == 'linear': return p['intercept'] + p['slope']*xx
        if model_id == 'gaussian': return p['baseline'] + p['amplitude']*np.exp(-.5*((xx-p['center'])/p['sigma'])**2)
        if model_id == 'lorentzian': return p['baseline'] + p['amplitude']/(1+((xx-p['center'])/p['gamma'])**2)
        if model_id == 'voigt': return p['baseline'] + p['amplitude']*voigt_profile(xx-p['center'], p['sigma'], p['gamma'])/voigt_profile(0., p['sigma'], p['gamma'])
        if model_id == 'exp1': return p['baseline'] + p['amplitude']*np.exp(p['rate']*xx)
        if model_id == 'exp2': return p['baseline'] + p['amplitude1']*np.exp(p['rate1']*xx) + p['amplitude2']*np.exp(p['rate2']*xx)
        if model_id == 'power_law': return p['baseline'] + p['amplitude']*xx**p['exponent']
        if model_id == 'stretched_exp': return p['baseline'] + p['amplitude']*np.exp(-(xx/p['tau'])**p['beta'])
        if model_id in ('logistic4', 'hill', 'logistic5'):
            midpoint = p['midpoint_scale'] if model_id == 'logistic5' else p['ec50']
            fraction = expit(p['hill']*(np.log(xx)-np.log(midpoint)))
            if model_id == 'logistic5': fraction = fraction**p['asymmetry']
            return p['bottom'] + (p['top']-p['bottom'])*fraction
        if model_id == 'boltzmann': return p['bottom'] + (p['top']-p['bottom'])*expit((xx-p['v_half'])/p['slope'])
    raise ValueError(f'Unsupported model: {model_id}.')


def _local_model_jacobian(evaluate, x, parameters, names, bounds, scales, cancel):
    """Physical-parameter sensitivities, independent of optimizer step scaling.

    A trust-region scale can be much larger than a fitted width or dose midpoint.
    Use local parameter/domain scales BEFORE choosing a central-difference step;
    merely clipping an oversized step at a bound biases the covariance. Location
    parameters receive an x-span scale from the caller. Finiteness alone is not
    accuracy: e.g. sin(a*x) at large a can cross many periods in that first step.
    Require agreement at three successively smaller steps. If floating-point
    resolution prevents convergence, refuse uncertainty instead of publishing a
    biased derivative (or an interval narrower than representable coordinates).
    """
    jacobian = np.empty((len(x), len(names)))
    relative_step = np.cbrt(np.finfo(float).eps)
    for i, name in enumerate(names):
        if cancel():
            raise FitCancelled('Curve fit cancelled.')
        value = parameters[name]
        low, high = bounds[name]
        room = min(value-low, high-value)
        scale = min(float(scales[i]), room)
        step = relative_step*scale
        # Ensure the two points remain representable at large coordinate origins.
        step = max(step, 32*abs(float(np.spacing(value))))
        if not np.isfinite(step) or step <= 0 or step >= room:
            raise ValueError(f'No resolvable local derivative for {name}.')
        previous = None
        agreements = 0
        for _ in range(24):
            if cancel():
                raise FitCancelled('Curve fit cancelled.')
            upper, lower = value+step, value-step
            if not low <= lower < value < upper <= high:
                break
            plus, minus = dict(parameters), dict(parameters)
            plus[name], minus[name] = upper, lower
            with np.errstate(all='ignore'):
                forward = np.asarray(evaluate(x, plus), dtype=float)
                backward = np.asarray(evaluate(x, minus), dtype=float)
                derivative = (forward-backward)/(upper-lower)
            if derivative.shape == x.shape and np.all(np.isfinite(derivative)):
                if previous is not None:
                    magnitude = max(float(np.max(np.abs(derivative))),
                                    float(np.max(np.abs(previous))), np.finfo(float).tiny)
                    error = float(np.max(np.abs(derivative-previous)))/magnitude
                    agreements = agreements+1 if error <= 1e-5 else 0
                    if agreements >= 2:
                        # The middle of the three agreeing estimates avoids the
                        # extra cancellation of the smallest validating step.
                        jacobian[:, i] = previous
                        break
                previous = derivative
            else:
                previous = None
                agreements = 0
            step *= .25
        else:
            raise ValueError(f'Local derivative for {name} did not converge to a finite reliable value.')
        if not low <= lower < value < upper <= high:
            raise ValueError(f'Floating-point resolution prevents a reliable local derivative for {name}; rescale or recenter the data/equation.')
    return jacobian


def fit_curve(x: Sequence[float], y: Sequence[float], config: Mapping[str, Any],
              y_sigma: Sequence[float] | None = None, cancel: Callable[[], bool] | None = None,
              *, cancelled: Callable[[], bool] | None = None) -> CurveFitResult:
    """Fit a declared model in original y space, with honest local uncertainty.

    ``bounds`` accepts ``None`` for infinity. ``fixed`` maps parameter names to
    values. A supplied sigma is a known observation standard deviation, so weighted
    covariance uses absolute sigma. Robust losses currently omit conventional
    least-squares intervals rather than mislabeling them as robust intervals.
    """
    check_cancel = cancel or cancelled or (lambda: False)
    if check_cancel(): raise FitCancelled('Curve fit cancelled.')
    xx, yy = _xy(x, y)
    cfg = dict(config)
    model_id = str(cfg.get('model', 'gaussian')); model = get_model(model_id)
    names = tuple(cfg.get('custom_parameters', ())) if model_id == 'custom' else model.parameter_names
    custom = compile_custom_expression(str(cfg.get('custom_expression', '')), names) if model_id == 'custom' else None
    evaluate = custom or (lambda z, p: evaluate_curve(model_id, z, p))
    lo_x = cfg.get('x_min'); hi_x = cfg.get('x_max')
    if lo_x is not None and not np.isfinite(float(lo_x)): raise ValueError('Lower x limit must be finite.')
    if hi_x is not None and not np.isfinite(float(hi_x)): raise ValueError('Upper x limit must be finite.')
    if lo_x is not None and hi_x is not None and float(lo_x) >= float(hi_x): raise ValueError('Lower x limit must be less than upper x limit.')
    for key in ('x_min', 'x_max'):
        if cfg.get(key) is not None:
            cfg[key] = float(cfg[key])
    n_input = int(xx.size)
    selected = np.ones(xx.size, dtype=bool)
    if lo_x is not None: selected &= xx >= float(lo_x)
    if hi_x is not None: selected &= xx <= float(hi_x)
    sigma = None
    if y_sigma is not None:
        sigma = np.asarray(y_sigma, dtype=float)
        if sigma.shape != xx.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError('Each observation uncertainty must be finite, positive and match x/y.')
        sigma = sigma[selected]
    xx, yy = _xy(xx[selected], yy[selected])
    if model.positive_x and np.any(xx < 0): raise ValueError(f'{model.label} requires non-negative x; use Boltzmann for signed voltage.')
    if model.strictly_positive_x and np.any(xx <= 0): raise ValueError(f'{model.label} requires strictly positive x.')
    weighting = str(cfg.get('weighting', 'uniform'))
    if weighting not in ('uniform', 'sigma'): raise ValueError('Weighting must be uniform or sigma.')
    if weighting == 'sigma' and sigma is None: raise ValueError('Sigma weighting requires observation standard deviations.')
    scale_y = sigma if weighting == 'sigma' else np.ones(xx.size)
    loss = str(cfg.get('loss', 'linear'))
    if loss not in ('linear', 'soft_l1', 'huber'): raise ValueError('Loss must be linear, soft_l1 or huber.')
    confidence = float(cfg.get('confidence', .95))
    if not .5 <= confidence < 1: raise ValueError('Confidence must be at least 0.5 and below 1.')
    initial = dict(cfg.get('initial') or {}); fixed = dict(cfg.get('fixed') or {}); bounds = dict(cfg.get('bounds') or {})
    for group in (initial, fixed, bounds):
        if set(group)-set(names): raise ValueError('Unknown model parameters: '+', '.join(sorted(set(group)-set(names)))+'.')
    guess = {p: 1. for p in names} if custom else model.guess(xx, yy)
    guess.update(initial); guess.update(fixed)
    bound_map = {p: (-np.inf, np.inf) for p in names} if custom else model.default_bounds(xx, yy)
    physical_positive = {p for p,(lo,hi) in bound_map.items() if lo > 0}
    for p, pair in bounds.items():
        if len(pair) != 2: raise ValueError(f'Bounds for {p} must have lower and upper values.')
        low = -np.inf if pair[0] is None else float(pair[0])
        high = np.inf if pair[1] is None else float(pair[1])
        if p in physical_positive:
            if low < 0: raise ValueError(f'{p} must remain positive; its lower bound cannot be negative or unbounded.')
            low = max(low,bound_map[p][0])
        bound_map[p] = (low,high)
    for p in names:
        guess[p] = float(guess[p]); low, high = bound_map[p]
        if np.isnan(low) or np.isnan(high) or low >= high: raise ValueError(f'Invalid bounds for {p}; use Fixed for a constant parameter.')
        if not np.isfinite(guess[p]): raise ValueError(f'Initial value for {p} must be finite.')
        if not low <= guess[p] <= high:
            if p in initial or p in fixed: raise ValueError(f'Initial/fixed value for {p} is outside its bounds.')
            guess[p] = float(np.clip(guess[p], low, high))
    fixed = {p: guess[p] for p in fixed}
    free = tuple(p for p in names if p not in fixed)
    n, k = len(xx), len(free)
    if n <= k: raise ValueError(f'{model.label} has {k} free parameters but only {n} observations. More than {k} observations are required; fix justified parameters or collect more data.')
    if len(np.unique(xx)) < max(2, k): raise ValueError('Too few distinct x values to identify the free model parameters.')
    trial = np.asarray(evaluate(xx, guess))
    if trial.shape != xx.shape or not np.all(np.isfinite(trial)):
        raise ValueError('The model is non-finite at the initial parameters. Check its domain, initial values and bounds.')
    warnings: list[str] = []
    sx = max(float(np.ptp(xx)), 1e-12); sy = max(float(np.ptp(yy)), float(np.std(yy)), abs(float(np.mean(yy)))*.01, 1e-100)
    # Optimize dimensionless steps around the initial point. This avoids forcing
    # tiny times and large currents into the same trust-region length scale.
    def param_scale(p: str) -> float:
        if p == 'slope' and model_id == 'linear': return max(abs(guess[p]), sy/sx)
        if p in ('intercept','baseline','amplitude','amplitude1','amplitude2','bottom','top'): return max(abs(guess[p]), sy)
        if p in ('center','v_half'): return sx
        if p in ('sigma','gamma','tau','ec50','midpoint_scale','slope'): return max(abs(guess[p]), sx*.01)
        if p in ('rate','rate1','rate2'): return max(abs(guess[p]), 1/sx)
        return max(abs(guess[p]), 1.)
    ps = np.array([param_scale(p) for p in free]); p0 = np.array([guess[p] for p in free])
    lower = np.array([(bound_map[p][0]-guess[p])/ps[i] for i,p in enumerate(free)])
    upper = np.array([(bound_map[p][1]-guess[p])/ps[i] for i,p in enumerate(free)])
    def unpack(z: np.ndarray) -> dict[str, float]:
        return {**guess, **{p: float(p0[i]+ps[i]*z[i]) for i,p in enumerate(free)}}
    def residual(z: np.ndarray) -> np.ndarray:
        if check_cancel(): raise FitCancelled('Curve fit cancelled.')
        predicted = np.asarray(evaluate(xx, unpack(z)), dtype=float)
        # Invalid trial points should backtrack, while invalid initial/final values
        # are errors. Finite penalty also handles user equations near a boundary.
        return np.nan_to_num((predicted-yy)/(scale_y if weighting == 'sigma' else sy), nan=1e50, posinf=1e50, neginf=-1e50)
    f_scale = float(cfg.get('f_scale', 1. if weighting == 'sigma' else sy*.1))
    if not np.isfinite(f_scale) or f_scale <= 0: raise ValueError('Robust residual scale must be finite and positive.')
    max_nfev = min(max(int(cfg.get('max_nfev', 5000)), 10), 50000)
    # Persist the options actually executed, including supported numeric coercion
    # and the evaluation-budget clamp, rather than an unreloadable raw spelling.
    if 'f_scale' in cfg:
        cfg['f_scale'] = f_scale
    if 'max_nfev' in cfg:
        cfg['max_nfev'] = max_nfev
    if k:
        opt = least_squares(residual, np.zeros(k), bounds=(lower, upper), loss=loss, f_scale=f_scale/(sy if weighting == 'uniform' else 1.),
                            x_scale='jac', jac='3-point', max_nfev=max_nfev,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        params = unpack(opt.x); success = bool(opt.success); message = str(opt.message)
        jac = opt.jac*(sy if weighting == 'uniform' else 1.)
        if not success: warnings.append('Optimizer did not converge: '+message)
    else:
        opt = None; params = guess; success = True; message = 'All parameters fixed; model evaluated without optimization.'
        jac = np.empty((n,0))
    if check_cancel(): raise FitCancelled('Curve fit cancelled.')
    y_fit = np.asarray(evaluate(xx, params), dtype=float)
    if not np.all(np.isfinite(y_fit)): raise ValueError('Fitted model is not finite over the observations.')
    res = yy-y_fit; sse = float(np.dot(res,res)); dof = n-k
    # Remove the origin before centering so a DC offset cannot erase an otherwise
    # resolved response variation. Half an ULP is the rounding uncertainty of an
    # observation; one-ULP jitter alone does not establish a resolved variance.
    shifted_y = yy-yy[0]
    centered_y = shifted_y-np.mean(shifted_y)
    tss = float(np.dot(centered_y, centered_y))
    response_resolution = .5*float(np.max(np.abs(np.spacing(yy))))
    r_squared = 1-sse/tss if tss > n*response_resolution**2 else None
    chi2 = float(np.sum((res/scale_y)**2)); mse = sse/dof
    x_curve = np.geomspace(float(np.min(xx[xx>0])), float(np.max(xx)), 400) if model_id in ('logistic4','logistic5','hill','power_law') and np.min(xx)>0 and np.max(xx)/np.min(xx)>100 else np.linspace(np.min(xx),np.max(xx),400)
    y_curve = np.asarray(evaluate(x_curve, params),dtype=float)
    if not np.all(np.isfinite(y_curve)): raise ValueError('Fitted model is not finite between observations; revise its domain or bounds.')
    se: dict[str,float|None] = {p: 0. if p in fixed else None for p in names}
    ci: dict[str,tuple[float,float]|None] = {p: (params[p],params[p]) if p in fixed else None for p in names}
    conf_lo = conf_hi = pred_lo = pred_hi = None
    rank = 0; condition = None; covariance_valid = False
    active_bounds = []
    for p in free:
        low,high=bound_map[p]; tolerance=1e-6*max(abs(params[p]),param_scale(p))
        if (np.isfinite(low) and params[p]-low <= tolerance) or (np.isfinite(high) and high-params[p] <= tolerance): active_bounds.append(p)
    if active_bounds: warnings.append('Parameters are at/near bounds: '+', '.join(active_bounds)+'. Symmetric local intervals are unreliable and have been omitted.')
    ambiguous_prediction_noise = False
    if weighting == 'sigma':
        order = np.argsort(xx, kind='stable')
        sorted_x, sorted_sigma = xx[order], sigma[order]
        ambiguous_prediction_noise = bool(np.any((np.diff(sorted_x) == 0) & (np.diff(sorted_sigma) != 0)))
        if ambiguous_prediction_noise:
            warnings.append('Prediction bands are omitted because repeated x values have different supplied observation standard deviations; a future-observation noise policy must be specified.')
        else:
            warnings.append('Prediction bands assume the supplied observation standard deviations also apply between observations, using linear interpolation in x.')
    if loss != 'linear': warnings.append('Robust loss was used. Conventional least-squares confidence/prediction intervals are omitted; use a suitable bootstrap for robust uncertainty.')
    if k:
        sensitivity_valid = True
        try:
            local_scales = [sx if p in ('center', 'v_half') else
                            abs(params[p]) if p in physical_positive else
                            max(abs(params[p]), ps[i]) for i, p in enumerate(free)]
            sensitivity = _local_model_jacobian(
                evaluate, np.concatenate((xx, x_curve)), params, free,
                bound_map, local_scales, check_cancel)
            gradient = sensitivity[n:]
            # Rank/conditioning use dimensionless optimizer coordinates, but the
            # derivatives themselves are evaluated in accurate local physical steps.
            jac = sensitivity[:n]*ps[None, :]/scale_y[:, None]
            if not np.all(np.isfinite(jac)):
                raise ValueError('Non-finite scaled sensitivity.')
        except ValueError as exc:
            sensitivity_valid = False
            warnings.append('Local uncertainty could not be evaluated reliably; '
                            'intervals have been omitted. '+str(exc))
        if sensitivity_valid:
            singular = np.linalg.svd(jac, compute_uv=False)
            rank = int(np.sum(singular > singular[0]*max(jac.shape)*np.finfo(float).eps)) if singular.size and singular[0]>0 else 0
            condition = float(singular[0]/singular[-1]) if singular.size and singular[-1]>0 else np.inf
            if rank < k or condition > 1e10:
                warnings.append('Parameters are not reliably identifiable (rank deficient or ill-conditioned Jacobian). Intervals have been omitted; simplify the model or expand the x range.')
        covariance_valid = sensitivity_valid and success and rank == k and condition <= 1e10 and loss == 'linear' and not active_bounds
        if covariance_valid:
            _u, _s, _vh = np.linalg.svd(jac, full_matrices=False)
            with np.errstate(all='ignore'):
                covariance_z = ((_vh.T/_s**2)@_vh)*(1. if weighting=='sigma' else chi2/dof)
                covariance = covariance_z*ps[:,None]*ps[None,:]
                quantile = float(norm.ppf((1+confidence)/2) if weighting=='sigma' else student_t.ppf((1+confidence)/2,dof))
                errors = np.sqrt(np.maximum(0, np.diag(covariance)))
                intervals = np.asarray([(params[p]-quantile*errors[i], params[p]+quantile*errors[i]) for i,p in enumerate(free)])
                variance=np.maximum(np.einsum('ij,jk,ik->i',gradient,covariance,gradient),0)
                delta=quantile*np.sqrt(variance)
                if weighting=='sigma' and not ambiguous_prediction_noise:
                    unique_x, unique_indices = np.unique(sorted_x, return_index=True)
                    observation_variance=np.interp(x_curve,unique_x,sorted_sigma[unique_indices])**2
                else: observation_variance=mse
                pred_delta=quantile*np.sqrt(variance+observation_variance)
                bands = (y_curve-delta, y_curve+delta)
                if not ambiguous_prediction_noise:
                    bands += (y_curve-pred_delta, y_curve+pred_delta)
            # Never label a partially non-finite diagnostic as a valid interval, or
            # emit a fit record that its own validated project loader cannot reopen.
            covariance_valid = all(np.all(np.isfinite(a)) for a in (covariance, errors, intervals, *bands))
            unresolved = [p for i, p in enumerate(free) if covariance_valid and errors[i] > 0 and
                          not intervals[i, 0] < params[p] < intervals[i, 1]]
            if covariance_valid and unresolved:
                covariance_valid = False
                warnings.append('Parameter intervals are narrower than floating-point resolution for '+', '.join(unresolved)+'. Local uncertainty has been omitted; rescale or recenter the data/equation.')
            if covariance_valid:
                for i,p in enumerate(free):
                    se[p] = float(errors[i]); ci[p] = tuple(float(v) for v in intervals[i])
                conf_lo, conf_hi = bands[:2]
                if not ambiguous_prediction_noise:
                    pred_lo, pred_hi = bands[2:]
            elif not unresolved:
                warnings.append('Local uncertainty is non-finite; intervals have been omitted. '
                                'Review parameter scales, bounds and model identifiability.')
    else:
        conf_lo=y_curve.copy();conf_hi=y_curve.copy()
        warnings.append('All parameters were fixed. The displayed curve is an evaluation, not an estimated fit.')
    if model_id in ('logistic4','logistic5','hill','boltzmann'):
        span_y=abs(params['top']-params['bottom'])
        if span_y>0 and np.ptp(y_curve)/span_y < .8:
            warnings.append('The observed x range does not span both sigmoid plateaus; asymptotes and midpoint may be weakly identified.')
    if model_id=='exp2' and abs(params['rate1']-params['rate2'])*sx < .1:
        warnings.append('The two exponential rates nearly coincide; component amplitudes/rates may not be individually identifiable.')
    if r_squared is None: warnings.append('R-squared is undefined for an effectively constant response.')
    if loss!='linear': aic=bic=None
    elif weighting=='sigma':
        log_likelihood=-.5*(chi2+float(np.sum(np.log(2*np.pi*sigma**2))))
        aic=2*k-2*log_likelihood;bic=k*np.log(n)-2*log_likelihood
    else:
        # Includes the fitted residual-variance parameter in information criteria.
        log_likelihood=-.5*n*(np.log(2*np.pi*max(sse/n,np.finfo(float).tiny))+1)
        aic=2*(k+1)-2*log_likelihood;bic=(k+1)*np.log(n)-2*log_likelihood
    cfg.update(model=model_id,initial=guess,bounds=bound_map,fixed=fixed,loss=loss,weighting=weighting,confidence=confidence)
    metrics=dict(n=n,n_input=n_input,n_excluded_range=n_input-n,n_parameters=k,dof=dof,sse=sse,rmse=math.sqrt(mse),r_squared=r_squared,
                 adjusted_r_squared=(1-(1-r_squared)*(n-1)/dof if r_squared is not None else None),
                 chi_squared=(chi2 if weighting=='sigma' else None),reduced_chi_squared=(chi2/dof if weighting=='sigma' else None),
                 aic=aic,bic=bic,jacobian_rank=rank,jacobian_condition=condition,
                 objective='original_y',prediction_noise=('unavailable: repeated x with heterogeneous supplied sigma' if ambiguous_prediction_noise else 'linearly interpolated supplied sigma' if weighting=='sigma' else 'constant estimated residual variance'),interval_method=('local linearized, pointwise' if covariance_valid else 'unavailable'),
                 nfev=int(opt.nfev) if opt is not None else 0)
    return CurveFitResult(params,se,ci,xx.copy(),yy.copy(),y_fit,res,x_curve,y_curve,conf_lo,conf_hi,pred_lo,pred_hi,
                          _safe_json(metrics),warnings,_safe_json(cfg),success,message,sigma.copy() if sigma is not None else None)
