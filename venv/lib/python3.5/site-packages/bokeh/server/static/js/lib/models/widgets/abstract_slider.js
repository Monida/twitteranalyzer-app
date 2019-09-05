"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var noUiSlider = require("nouislider");
var p = require("../../core/properties");
var dom_1 = require("../../core/dom");
var array_1 = require("../../core/util/array");
var callback_1 = require("../../core/util/callback");
var control_1 = require("./control");
var sliders_1 = require("../../styles/widgets/sliders");
var prefix = 'bk-noUi-';
var AbstractBaseSliderView = /** @class */ (function (_super) {
    tslib_1.__extends(AbstractBaseSliderView, _super);
    function AbstractBaseSliderView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Object.defineProperty(AbstractBaseSliderView.prototype, "noUiSlider", {
        get: function () {
            return this.slider_el.noUiSlider;
        },
        enumerable: true,
        configurable: true
    });
    AbstractBaseSliderView.prototype.initialize = function () {
        _super.prototype.initialize.call(this);
        this._init_callback();
    };
    AbstractBaseSliderView.prototype.connect_signals = function () {
        var _this = this;
        _super.prototype.connect_signals.call(this);
        var _a = this.model.properties, callback = _a.callback, callback_policy = _a.callback_policy, callback_throttle = _a.callback_throttle;
        this.on_change([callback, callback_policy, callback_throttle], function () { return _this._init_callback(); });
        var _b = this.model.properties, start = _b.start, end = _b.end, value = _b.value, step = _b.step, title = _b.title;
        this.on_change([start, end, value, step], function () {
            var _a = _this._calc_to(), start = _a.start, end = _a.end, value = _a.value, step = _a.step;
            _this.noUiSlider.updateOptions({
                range: { min: start, max: end },
                start: value,
                step: step,
            });
        });
        var bar_color = this.model.properties.bar_color;
        this.on_change(bar_color, function () {
            _this._set_bar_color();
        });
        this.on_change([value, title], function () { return _this._update_title(); });
    };
    AbstractBaseSliderView.prototype._init_callback = function () {
        var _this = this;
        var callback = this.model.callback;
        var fn = function () {
            if (callback != null)
                callback.execute(_this.model);
            _this.model.value_throttled = _this.model.value;
        };
        switch (this.model.callback_policy) {
            case 'continuous': {
                this.callback_wrapper = fn;
                break;
            }
            case 'throttle': {
                this.callback_wrapper = callback_1.throttle(fn, this.model.callback_throttle);
                break;
            }
            default:
                this.callback_wrapper = undefined;
        }
    };
    AbstractBaseSliderView.prototype._update_title = function () {
        var _this = this;
        dom_1.empty(this.title_el);
        var hide_header = this.model.title == null || (this.model.title.length == 0 && !this.model.show_value);
        this.title_el.style.display = hide_header ? "none" : "";
        if (!hide_header) {
            if (this.model.title.length != 0)
                this.title_el.textContent = this.model.title + ": ";
            if (this.model.show_value) {
                var value = this._calc_to().value;
                var pretty = value.map(function (v) { return _this.model.pretty(v); }).join(" .. ");
                this.title_el.appendChild(dom_1.span({ class: sliders_1.bk_slider_value }, pretty));
            }
        }
    };
    AbstractBaseSliderView.prototype._set_bar_color = function () {
        if (!this.model.disabled) {
            this.slider_el.querySelector("." + prefix + "connect")
                .style
                .backgroundColor = this.model.bar_color;
        }
    };
    AbstractBaseSliderView.prototype._keypress_handle = function (e, idx) {
        if (idx === void 0) { idx = 0; }
        var _a = this._calc_to(), start = _a.start, value = _a.value, end = _a.end, step = _a.step;
        var is_range = value.length == 2;
        var low = start;
        var high = end;
        if (is_range && idx == 0) {
            high = value[1];
        }
        else if (is_range && idx == 1) {
            low = value[0];
        }
        switch (e.which) {
            case 37: {
                value[idx] = Math.max(value[idx] - step, low);
                break;
            }
            case 39: {
                value[idx] = Math.min(value[idx] + step, high);
                break;
            }
            default:
                return;
        }
        if (is_range) {
            this.model.value = value;
            this.model.properties.value.change.emit();
        }
        else {
            this.model.value = value[0];
        }
        this.noUiSlider.set(value);
        if (this.callback_wrapper != null)
            this.callback_wrapper();
    };
    AbstractBaseSliderView.prototype.render = function () {
        var _this = this;
        _super.prototype.render.call(this);
        var _a = this._calc_to(), start = _a.start, end = _a.end, value = _a.value, step = _a.step;
        var tooltips; // XXX
        if (this.model.tooltips) {
            var formatter = {
                to: function (value) { return _this.model.pretty(value); },
            };
            tooltips = array_1.repeat(formatter, value.length);
        }
        else
            tooltips = false;
        if (this.slider_el == null) {
            this.slider_el = dom_1.div();
            noUiSlider.create(this.slider_el, {
                cssPrefix: prefix,
                range: { min: start, max: end },
                start: value,
                step: step,
                behaviour: this.model.behaviour,
                connect: this.model.connected,
                tooltips: tooltips,
                orientation: this.model.orientation,
                direction: this.model.direction,
            }); // XXX: bad typings; no cssPrefix
            this.noUiSlider.on('slide', function (_, __, values) { return _this._slide(values); });
            this.noUiSlider.on('change', function (_, __, values) { return _this._change(values); });
            this._set_keypress_handles();
            var toggleTooltip_1 = function (i, show) {
                if (!tooltips)
                    return;
                var handle = _this.slider_el.querySelectorAll("." + prefix + "handle")[i];
                var tooltip = handle.querySelector("." + prefix + "tooltip");
                tooltip.style.display = show ? 'block' : '';
            };
            this.noUiSlider.on('start', function (_, i) { return toggleTooltip_1(i, true); });
            this.noUiSlider.on('end', function (_, i) { return toggleTooltip_1(i, false); });
        }
        else {
            this.noUiSlider.updateOptions({
                range: { min: start, max: end },
                start: value,
                step: step,
            });
        }
        this._set_bar_color();
        if (this.model.disabled)
            this.slider_el.setAttribute('disabled', 'true');
        else
            this.slider_el.removeAttribute('disabled');
        this.title_el = dom_1.div({ class: sliders_1.bk_slider_title });
        this._update_title();
        this.group_el = dom_1.div({ class: sliders_1.bk_input_group }, this.title_el, this.slider_el);
        this.el.appendChild(this.group_el);
    };
    AbstractBaseSliderView.prototype._slide = function (values) {
        this.model.value = this._calc_from(values);
        if (this.callback_wrapper != null)
            this.callback_wrapper();
    };
    AbstractBaseSliderView.prototype._change = function (values) {
        this.model.value = this._calc_from(values);
        this.model.value_throttled = this.model.value;
        switch (this.model.callback_policy) {
            case 'mouseup':
            case 'throttle': {
                if (this.model.callback != null)
                    this.model.callback.execute(this.model);
                break;
            }
        }
    };
    AbstractBaseSliderView.__name__ = "AbstractBaseSliderView";
    return AbstractBaseSliderView;
}(control_1.ControlView));
var AbstractSliderView = /** @class */ (function (_super) {
    tslib_1.__extends(AbstractSliderView, _super);
    function AbstractSliderView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    AbstractSliderView.prototype._calc_to = function () {
        return {
            start: this.model.start,
            end: this.model.end,
            value: [this.model.value],
            step: this.model.step,
        };
    };
    AbstractSliderView.prototype._calc_from = function (_a) {
        var value = _a[0];
        if (Number.isInteger(this.model.start) && Number.isInteger(this.model.end) && Number.isInteger(this.model.step))
            return Math.round(value);
        else
            return value;
    };
    AbstractSliderView.prototype._set_keypress_handles = function () {
        var _this = this;
        // Add single cursor event
        var handle = this.slider_el.querySelector("." + prefix + "handle");
        handle.setAttribute('tabindex', '0');
        handle.addEventListener('keydown', function (e) { return _this._keypress_handle(e); });
    };
    AbstractSliderView.__name__ = "AbstractSliderView";
    return AbstractSliderView;
}(AbstractBaseSliderView));
exports.AbstractSliderView = AbstractSliderView;
var AbstractRangeSliderView = /** @class */ (function (_super) {
    tslib_1.__extends(AbstractRangeSliderView, _super);
    function AbstractRangeSliderView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    AbstractRangeSliderView.prototype._calc_to = function () {
        return {
            start: this.model.start,
            end: this.model.end,
            value: this.model.value,
            step: this.model.step,
        };
    };
    AbstractRangeSliderView.prototype._calc_from = function (values) {
        return values;
    };
    AbstractRangeSliderView.prototype._set_keypress_handles = function () {
        var _this = this;
        var handle_lower = this.slider_el.querySelector("." + prefix + "handle-lower");
        var handle_upper = this.slider_el.querySelector("." + prefix + "handle-upper");
        handle_lower.setAttribute('tabindex', '0');
        handle_lower.addEventListener('keydown', function (e) { return _this._keypress_handle(e, 0); });
        handle_upper.setAttribute('tabindex', '1');
        handle_upper.addEventListener('keydown', function (e) { return _this._keypress_handle(e, 1); });
    };
    AbstractRangeSliderView.__name__ = "AbstractRangeSliderView";
    return AbstractRangeSliderView;
}(AbstractBaseSliderView));
exports.AbstractRangeSliderView = AbstractRangeSliderView;
var AbstractSlider = /** @class */ (function (_super) {
    tslib_1.__extends(AbstractSlider, _super);
    function AbstractSlider(attrs) {
        var _this = _super.call(this, attrs) || this;
        _this.connected = false;
        return _this;
    }
    AbstractSlider.initClass = function () {
        this.define({
            title: [p.String, ""],
            show_value: [p.Boolean, true],
            start: [p.Any],
            end: [p.Any],
            value: [p.Any],
            value_throttled: [p.Any],
            step: [p.Number, 1],
            format: [p.String],
            direction: [p.Any, "ltr"],
            tooltips: [p.Boolean, true],
            callback: [p.Any],
            callback_throttle: [p.Number, 200],
            callback_policy: [p.SliderCallbackPolicy, "throttle"],
            bar_color: [p.Color, "#e6e6e6"],
        });
    };
    AbstractSlider.prototype._formatter = function (value, _format) {
        return "" + value;
    };
    AbstractSlider.prototype.pretty = function (value) {
        return this._formatter(value, this.format);
    };
    AbstractSlider.__name__ = "AbstractSlider";
    return AbstractSlider;
}(control_1.Control));
exports.AbstractSlider = AbstractSlider;
AbstractSlider.initClass();
