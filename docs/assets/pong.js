var FRICTION = 0.6;
var VX = 0.03;
var VY0 = 0.01;
var VY1 = 0.05;
var TOP = 1.0;
var BOTTOM = -1.0;
var LEFT = -1.0;
var RIGHT = 1.0;
var HPL = 0.2;
var MAX_STEPS = 1000;
var DY = [0.0, -0.05, 0.05];

var canvas;
var context;

function Paddle(y, vy) {
    this.y = y;
    this.vy = vy;
}

function State(ball_x, ball_y, ball_vx, ball_vy, l_y, l_vy, r_y, r_vy) {
    this.ball_x = ball_x;
    this.ball_y = ball_y;
    this.ball_vx = ball_vx;
    this.ball_vy = ball_vy;
    this.l = new Paddle(l_y, l_vy);
    this.r = new Paddle(r_y, r_vy);
}

function Pong(max_steps) {
    this.max_steps = max_steps;
    this.reset();
}

Pong.prototype.set_state = function(state) {
    this.s = state;
    this.done = false;
    this.done = null;
    this.win = null;
    this.hit = null;
    this.miss = null;
    this.n_steps = 0;
};

Pong.prototype.reset = function() {
    this.s = new State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    this.score = {"l": 0, "r": 0, "draw": 0};
    this.new_episode();
};

function random_sign() {
    if (Math.random() >= 0.5) {
        return 1.0;
    } else {
        return -1.0;
    }    
}

function random_range(min, max) {
    return Math.random() * (max - min) + min;
}

Pong.prototype.new_episode = function() {
    this.s.ball_x = 0.0;
    this.s.ball_y = 0.0;
    this.s.ball_vx = random_sign() * VX;
    this.s.ball_vy = random_sign() * random_range(VY0, VY1);
    this.set_state(this.s);
};

Pong.prototype.step_paddle = function(p, a) {
    p.vy = (p.vy + DY[a]) * FRICTION;
    p.y = p.y + p.vy;
    
    if (p.y + HPL >= TOP) {
        p.y = TOP - HPL;
        p.vy = 0.0;
    }
    if (p.y - HPL <= BOTTOM) {
        p.y = BOTTOM + HPL;
        p.vy = 0.0;
    }   
};

Pong.prototype.step_ball = function() {
    
    var tt_x = Infinity;
    if (this.s.ball_vx > 0.0) {
        tt_x = (RIGHT - this.s.ball_x) / this.s.ball_vx;
    } else if (this.s.ball_vx < 0.0) {
        tt_x = (LEFT - this.s.ball_x) / this.s.ball_vx;
    }
    
    var tt_y = Infinity;
    if (this.s.ball_vy > 0.0) {
        tt_y = (RIGHT - this.s.ball_y) / this.s.ball_vy;
    } else if (this.s.ball_vy < 0.0) {
        tt_y = (LEFT - this.s.ball_y) / this.s.ball_vy;
    }
    
    if ((tt_x <= tt_y) && (tt_y <= 1.0)) {
        this.advance_ball(tt_x);
        this.hit_x();
        this.advance_ball(tt_y - tt_x);
        this.hit_y();
        this.advance_ball(1.0 - tt_y);
    } else if ((tt_y < tt_x) && (tt_x <= 1.0)) {
        this.advance_ball(tt_y);
        this.hit_y();
        this.advance_ball(tt_x - tt_y);
        this.hit_x();
        this.advance_ball(1.0 - tt_x);
    } else if (tt_x <= 1.0) {
        this.advance_ball(tt_x);
        this.hit_x();
        this.advance_ball(1.0 - tt_x);
    } else if (tt_y <= 1.0) {
        this.advance_ball(tt_y);
        this.hit_y();
        this.advance_ball(1.0 - tt_y);
    } else {
        this.advance_ball(1.0);
    }
};


Pong.prototype.advance_ball = function(t) {
    this.s.ball_x += t * this.s.ball_vx;
    this.s.ball_y += t * this.s.ball_vy;
};

Pong.prototype.hit_y = function() {
    this.s.ball_vy *= -1.0;
};

Pong.prototype.hit_x = function() {
    var p, side;
    if ((this.s.ball_x - LEFT) < Number.EPSILON) {
        p = this.s.l;
        side = "l";
    } else {
        p = this.s.r;
        side = "r";
    }
    if ((p.y - HPL < this.s.ball_y) && (this.s.ball_y < p.y + HPL)) {
        this.s.ball_vx *= -1.0;
        this.s.ball_vy += p.vy;
    } else {
        this.miss = side;
        this.win = (side == "r") ? "l" : "r";
        this.score[this.win] += 1;
        this.done = true;
    }
};

Pong.prototype.step = function(l_a, r_a) {
    this.step_paddle(this.s.l, l_a);
    this.step_paddle(this.s.r, r_a);
    this.step_ball();
    
    this.n_steps += 1;
    if ((!this.done) && (this.n_steps >= this.max_steps)) {
        this.done = true;
        this.score["draw"] += 1;
    }
};

/* ---------------------------------------------------
   ===================================================
   ---------------------------------------------------
*/

var animate = (window.requestAnimationFrame ||
               window.webkitRequestAnimationFrame ||
               window.mozRequestAnimationFrame ||
               function (callback) { window.setTimeout(callback, 1000 / 60)});
var keysDown = {};
var touchDown = {};

function PongGUI(canvas) {
    this.WIDTH = 1.0;
    this.HEIGHT = 1.0;
    this.PADDLE_HEIGHT = HPL * this.HEIGHT;
    this.PADDLE_WIDTH = 0.03 * this.WIDTH / 2;
    this.BALL_RADIUS = 0.03 * this.WIDTH / 2;
    this.TRUE_WIDTH = this.WIDTH + 2 * this.BALL_RADIUS + 2 * this.PADDLE_WIDTH;
    this.TRUE_HEIGHT = this.HEIGHT + 2 * this.BALL_RADIUS;

    this.sim = new Pong(MAX_STEPS);
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.resize();
    
}

PongGUI.prototype.resize = function() {
    var w = Math.min(this.canvas.parentElement.offsetWidth * 0.8, 400);
    this.canvas.width = w;
    this.canvas.height = (w / this.TRUE_WIDTH) * this.TRUE_HEIGHT;
    this.render();
};

PongGUI.prototype.update = function() {
    if (this.sim.done) {
        this.sim.new_episode();
    }

    var r_a = 0;
    for (var key in keysDown) {
        if (key == "ArrowDown") {
            r_a = 2;
        } else if (key == "ArrowUp") {
            r_a = 1;
        }
    }
    
    if (touchDown.x) {
        var r = touchDown.y / this.canvas.height
        var s = (this.sim.s.r.y - BOTTOM) / 2;
        if (r > s) {
            r_a = 2;
        } else if (r < s) {
            r_a = 1;
        }
    }
    
    var l_a = 0;
    var low = this.sim.s.l.y - 0.5 * HPL;
    var high = this.sim.s.l.y + 0.5 * HPL;
    
    if (this.sim.s.ball_y < low) {
        l_a = 1;
    } else if (this.sim.s.ball_y > high) {
        l_a = 2;
    }
    
    this.sim.step(l_a, r_a);
};

PongGUI.prototype.hr2g = function(r) {
    return ((r + 1.0) / 2.0) + this.BALL_RADIUS;
};

PongGUI.prototype.wr2g = function(r) {
    return ((r + 1.0) / 2.0) + this.BALL_RADIUS + this.PADDLE_WIDTH;
};

PongGUI.prototype.f = function(r) {
    return (r / this.TRUE_WIDTH) * this.canvas.width;
};

PongGUI.prototype.g = function(r) {
    return (r / this.TRUE_HEIGHT) * this.canvas.height;
};

PongGUI.prototype.render = function() {

    this.context.fillStyle = "#000000";
    this.context.fillRect(0, 0, this.f(this.TRUE_WIDTH), this.g(this.TRUE_HEIGHT));

    this.context.fillStyle = "#FF0000";
    this.context.fillRect(
        this.f(this.wr2g(LEFT) - this.PADDLE_WIDTH - this.BALL_RADIUS) ,
        this.g(this.hr2g(this.sim.s.l.y) - 0.5 * this.PADDLE_HEIGHT),
        this.f(this.PADDLE_WIDTH),
        this.g(this.PADDLE_HEIGHT));

    this.context.fillStyle = "#FF0000";
    this.context.fillRect(
        this.f(this.wr2g(RIGHT) + this.BALL_RADIUS),
        this.g(this.hr2g(this.sim.s.r.y) - 0.5 * this.PADDLE_HEIGHT),
        this.f(this.PADDLE_WIDTH),
        this.g(this.PADDLE_HEIGHT));

    this.context.beginPath();
    this.context.arc(
        this.f(this.wr2g(this.sim.s.ball_x)),
        this.g(this.hr2g(this.sim.s.ball_y)),
        this.f(this.BALL_RADIUS),
        2 * Math.PI, false);
    this.context.fillStyle = "#00FF00";
    this.context.fill();
    
    this.context.font = "bold 15px sans-serif";
    this.context.textBaseline = "bottom";
    this.context.textAlign = "start";
    this.context.fillStyle = "#ADD8E6";
    this.context.fillText("Follow", 0, this.g(this.TRUE_HEIGHT));
    
    this.context.textAlign = "end";
    this.context.fillText("Player",
        this.f(this.TRUE_WIDTH),
        this.g(this.TRUE_HEIGHT));
    
    score_string = this.sim.score["l"] + "|" + this.sim.score["draw"] + "|" + this.sim.score["r"];
    this.context.textAlign = "center";
    this.context.fillStyle = "#FFFF00";
    this.context.fillText(score_string,
        this.f(this.TRUE_WIDTH / 2),
        this.g(this.TRUE_HEIGHT));
    
    if (!running) {
        this.context.fillStyle = "rgba(255, 255, 255, 0.7)";
        this.context.fillRect(0, 0,
            this.f(this.TRUE_WIDTH),
            this.g(this.TRUE_HEIGHT));
        
        this.context.font = "bold 20px sans-serif";
        this.context.textBaseline = "center";
        this.context.fillStyle = "#3333AA";
        this.context.fillText("Click to Play",
            this.f(this.TRUE_WIDTH / 2),
            this.g(this.TRUE_HEIGHT / 2));
    }
};

var now, then = 0.0;
var running = false;
var gui;

function step() {
    now = Date.now();
    elapsed = now - then;
    
    if (elapsed > 20.0) {
        gui.update();
        gui.render();
        then = now;
    }
    if (running) {
        animate(this.step);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    canvas = document.getElementById("pong-canvas");
    if (canvas) {
        gui = new PongGUI(canvas);
        var lastDownTarget;
        
        document.addEventListener('mousedown', function(event) {
            if ((lastDownTarget == canvas) && (event.target != canvas)) {
                running = false;
                gui.render();
            } else if ((lastDownTarget != canvas) && (event.target == canvas)) {
                running = true;
                animate(step);
            }
            lastDownTarget = event.target;
        }, false);

        document.addEventListener('keydown', function(event) {
            if (lastDownTarget == canvas) {
                keysDown[event.key] = true;
                event.preventDefault();
            }
        }, false);

        document.addEventListener('keyup', function(event) {
            if (lastDownTarget == canvas) {
                delete keysDown[event.key];
                event.preventDefault();
            }
        }, false);
        
        canvas.addEventListener("touchstart", function(event) {
            if (lastDownTarget != canvas) {
                running = true;
                animate(step);
                lastDownTarget = canvas;
            }
            if (event.touches) {
                touchDown["x"] = event.touches[0].pageX - canvas.offsetLeft;
                touchDown["y"] = event.touches[0].pageY - canvas.offsetTop;
                event.preventDefault();
            }
        }, false);
        canvas.addEventListener("touchmove", function(event) {
            if (lastDownTarget != canvas) {
                running = true;
                animate(step);
                lastDownTarget = canvas;
            }
            if (event.touches) {
                touchDown["x"] = event.touches[0].pageX - canvas.offsetLeft;
                touchDown["y"] = event.touches[0].pageY - canvas.offsetTop;
                event.preventDefault();
            }
        }, false);
        canvas.addEventListener("touchend", function(event) {
            delete touchDown["x"];
            delete touchDown["y"];
            event.preventDefault();
        }, false);
        canvas.addEventListener("touchcancel", function(event) {
            delete touchDown["x"];
            delete touchDown["y"];
            event.preventDefault();
        }, false);
        window.addEventListener('resize', function(event) {
            gui.resize();
        }, false);
    }
}, false);
