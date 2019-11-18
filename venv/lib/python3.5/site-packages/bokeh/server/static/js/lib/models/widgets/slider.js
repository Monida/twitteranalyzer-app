"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var numbro = require("numbro");
var abstract_slider_1 = require("./abstract_slider");
var SliderView = /** @class */ (function (_super) {
    tslib_1.__extends(SliderView, _super);
    function SliderView() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    SliderView.__name__ = "SliderView";
    return SliderView;
}(abstract_slider_1.AbstractSliderView));
exports.SliderView = SliderView;
var Slider = /** @class */ (function (_super) {
    tslib_1.__extends(Slider, _super);
    function Slider(attrs) {
        var _this = _super.call(this, attrs) || this;
        _this.behaviour = "tap";
        _this.connected = [true, false];
        return _this;
    }
    Slider.initClass = function () {
        this.prototype.default_view = SliderView;
        this.override({
            format: "0[.]00",
        });
    };
    Slider.prototype._formatter = function (value, format) {
        return numbro.format(value, format);
    };
    Slider.__name__ = "Slider";
    return Slider;
}(abstract_slider_1.AbstractSlider));
exports.Slider = Slider;
Slider.initClass();
