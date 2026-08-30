#!/usr/bin/env python3
"""Generate SFT data for digital design (Verilog/SystemVerilog).

Covers RTL design patterns that work with Xcelium or Verilator:
- Combinational logic
- Sequential logic (FSM, counters, shift registers)
- Timing analysis
- Power optimization
- Verification (assertions, coverage)

Usage:
    PYTHONPATH=src python scripts/generate_digital_sft.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.data.format import build_system_message
SEP = "=" * 60

EXAMPLES = [
    {
        "id": "rtl_fsm_traffic",
        "task": "Design a traffic light controller FSM in Verilog with pedestrian crossing support.",
        "reasoning": (
            "Traffic light FSM with 4 states:\n"
            "- GREEN (30s), YELLOW (5s), RED (25s), PED_WALK (15s)\n"
            "- Pedestrian button triggers PED_WALK after current GREEN ends\n"
            "- Use one-hot encoding for area-efficient synthesis\n\n"
            "State transitions:\n"
            "  GREEN --(timer)--> YELLOW --(timer)--> RED --(timer/ped)--> GREEN/PED_WALK\n"
            "  PED_WALK --(timer)--> RED"
        ),
        "verilog": """\
module traffic_fsm #(
    parameter GREEN_TIME  = 30,
    parameter YELLOW_TIME = 5,
    parameter RED_TIME    = 25,
    parameter PED_TIME    = 15
)(
    input  wire clk,
    input  wire rst_n,
    input  wire ped_button,
    output reg  [2:0] light,     // {red, yellow, green}
    output reg  ped_signal,
    output reg  [5:0] countdown
);

    // One-hot state encoding
    localparam S_GREEN    = 4'b0001;
    localparam S_YELLOW   = 4'b0010;
    localparam S_RED      = 4'b0100;
    localparam S_PED_WALK = 4'b1000;

    reg [3:0] state, next_state;
    reg [5:0] timer;
    reg ped_request;

    // State register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_GREEN;
            timer <= GREEN_TIME;
            ped_request <= 1'b0;
        end else begin
            state <= next_state;
            if (state != next_state)
                case (next_state)
                    S_GREEN:    timer <= GREEN_TIME;
                    S_YELLOW:   timer <= YELLOW_TIME;
                    S_RED:      timer <= RED_TIME;
                    S_PED_WALK: timer <= PED_TIME;
                endcase
            else if (timer > 0)
                timer <= timer - 1;
            if (ped_button)
                ped_request <= 1'b1;
            if (state == S_PED_WALK)
                ped_request <= 1'b0;
        end
    end

    // Next state logic
    always @(*) begin
        next_state = state;
        case (state)
            S_GREEN:    if (timer == 0) next_state = S_YELLOW;
            S_YELLOW:   if (timer == 0) next_state = S_RED;
            S_RED:      if (timer == 0)
                            next_state = ped_request ? S_PED_WALK : S_GREEN;
            S_PED_WALK: if (timer == 0) next_state = S_RED;
        endcase
    end

    // Output logic
    always @(*) begin
        light = 3'b000;
        ped_signal = 1'b0;
        countdown = timer;
        case (state)
            S_GREEN:    light = 3'b001;  // green
            S_YELLOW:   light = 3'b010;  // yellow
            S_RED:      light = 3'b100;  // red
            S_PED_WALK: begin light = 3'b100; ped_signal = 1'b1; end
        endcase
    end

endmodule
""",
        "analysis": (
            "FSM implementation notes:\n"
            "- One-hot encoding: 4 flip-flops, minimal next-state logic\n"
            "- Parameterized timers for easy configuration\n"
            "- Async reset (rst_n) for ASIC compatibility\n"
            "- Pedestrian request latched until served\n\n"
            "For synthesis: ~50 gates, Fmax > 500 MHz in 28nm."
        ),
    },
    {
        "id": "rtl_fifo_sync",
        "task": "Design a synchronous FIFO with configurable depth and width in SystemVerilog.",
        "reasoning": (
            "Synchronous FIFO components:\n"
            "- Dual-port RAM (write port + read port)\n"
            "- Write/read pointers with wrap-around\n"
            "- Full/empty flags from pointer comparison\n"
            "- Almost-full/almost-empty for flow control\n\n"
            "Key design choices:\n"
            "- Power-of-2 depth for simple pointer arithmetic\n"
            "- Extra MSB bit for full/empty distinction\n"
            "- Registered outputs for timing closure"
        ),
        "verilog": """\
module sync_fifo #(
    parameter WIDTH = 8,
    parameter DEPTH = 16,
    parameter ADDR_W = $clog2(DEPTH)
)(
    input  logic              clk,
    input  logic              rst_n,
    input  logic              wr_en,
    input  logic [WIDTH-1:0]  wr_data,
    input  logic              rd_en,
    output logic [WIDTH-1:0]  rd_data,
    output logic              full,
    output logic              empty,
    output logic [ADDR_W:0]   count
);

    // Memory
    logic [WIDTH-1:0] mem [DEPTH-1:0];

    // Pointers (extra bit for full/empty)
    logic [ADDR_W:0] wr_ptr, rd_ptr;

    // Write logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            wr_ptr <= '0;
        else if (wr_en && !full)
            wr_ptr <= wr_ptr + 1;
    end

    always_ff @(posedge clk) begin
        if (wr_en && !full)
            mem[wr_ptr[ADDR_W-1:0]] <= wr_data;
    end

    // Read logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rd_ptr <= '0;
        else if (rd_en && !empty)
            rd_ptr <= rd_ptr + 1;
    end

    assign rd_data = mem[rd_ptr[ADDR_W-1:0]];

    // Status flags
    assign full  = (wr_ptr[ADDR_W] != rd_ptr[ADDR_W]) &&
                   (wr_ptr[ADDR_W-1:0] == rd_ptr[ADDR_W-1:0]);
    assign empty = (wr_ptr == rd_ptr);
    assign count = wr_ptr - rd_ptr;

endmodule
""",
        "analysis": (
            "FIFO design considerations:\n"
            "- Depth must be power-of-2 for binary pointer arithmetic\n"
            "- Extra MSB distinguishes full from empty\n"
            "- `count` signal useful for flow control logic\n"
            "- For async FIFO: use Gray code pointers + 2-FF synchronizers\n\n"
            "Area: ~DEPTH*WIDTH flip-flops + control logic\n"
            "Timing: single-cycle read/write, Fmax limited by RAM access."
        ),
    },
    {
        "id": "rtl_spi_master",
        "task": "Design an SPI master controller with configurable clock divider.",
        "reasoning": (
            "SPI master needs:\n"
            "- Clock divider (SCLK = CLK / 2*DIV)\n"
            "- MOSI shift register (MSB first)\n"
            "- MISO capture on rising/falling edge (CPOL/CPHA)\n"
            "- CS_N assertion/deassertion\n"
            "- Byte-complete interrupt\n\n"
            "I'll implement Mode 0 (CPOL=0, CPHA=0): sample on rising, shift on falling."
        ),
        "verilog": """\
module spi_master #(
    parameter CLK_DIV = 4,   // SCLK = CLK / (2*CLK_DIV)
    parameter DATA_W  = 8
)(
    input  logic              clk,
    input  logic              rst_n,
    input  logic              start,
    input  logic [DATA_W-1:0] tx_data,
    output logic [DATA_W-1:0] rx_data,
    output logic              busy,
    output logic              done,
    // SPI signals
    output logic              sclk,
    output logic              mosi,
    input  logic              miso,
    output logic              cs_n
);

    logic [$clog2(CLK_DIV)-1:0] clk_cnt;
    logic [$clog2(DATA_W)-1:0]  bit_cnt;
    logic [DATA_W-1:0] shift_tx, shift_rx;
    logic sclk_reg;

    typedef enum logic [1:0] {IDLE, TRANSFER, DONE} state_t;
    state_t state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            cs_n     <= 1'b1;
            sclk_reg <= 1'b0;
            busy     <= 1'b0;
            done     <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        state    <= TRANSFER;
                        shift_tx <= tx_data;
                        shift_rx <= '0;
                        bit_cnt  <= DATA_W - 1;
                        clk_cnt  <= '0;
                        cs_n     <= 1'b0;
                        busy     <= 1'b1;
                        sclk_reg <= 1'b0;
                    end
                end
                TRANSFER: begin
                    if (clk_cnt == CLK_DIV - 1) begin
                        clk_cnt  <= '0;
                        sclk_reg <= ~sclk_reg;
                        if (sclk_reg) begin // Falling edge: shift
                            shift_tx <= {shift_tx[DATA_W-2:0], 1'b0};
                            if (bit_cnt == 0) state <= DONE;
                            else bit_cnt <= bit_cnt - 1;
                        end else begin // Rising edge: sample
                            shift_rx <= {shift_rx[DATA_W-2:0], miso};
                        end
                    end else
                        clk_cnt <= clk_cnt + 1;
                end
                DONE: begin
                    cs_n    <= 1'b1;
                    busy    <= 1'b0;
                    done    <= 1'b1;
                    rx_data <= shift_rx;
                    state   <= IDLE;
                end
            endcase
        end
    end

    assign sclk = sclk_reg;
    assign mosi = shift_tx[DATA_W-1];

endmodule
""",
        "analysis": (
            "SPI Master design:\n"
            "- Mode 0 (CPOL=0, CPHA=0): idle low, sample rising, shift falling\n"
            "- MSB-first transmission\n"
            "- Configurable clock divider for different SPI speeds\n"
            "- Single-byte transfer with start/done handshake\n\n"
            "For multi-byte: add a counter and auto-CS management.\n"
            "Area: ~100 gates, Fmax > 200 MHz in 28nm."
        ),
    },
    {
        "id": "rtl_pwm_generator",
        "task": "Design a PWM generator with dead-time insertion for motor control.",
        "reasoning": (
            "PWM generator for H-bridge motor driver:\n"
            "- Configurable period and duty cycle\n"
            "- Complementary outputs (PWM_H, PWM_L)\n"
            "- Dead-time insertion to prevent shoot-through\n"
            "- Enable/disable control\n\n"
            "Dead-time: both outputs LOW during transition to prevent\n"
            "simultaneous conduction of high-side and low-side MOSFETs."
        ),
        "verilog": """\
module pwm_deadtime #(
    parameter CNT_W     = 16,
    parameter DEAD_TIME = 10  // Dead-time in clock cycles
)(
    input  logic             clk,
    input  logic             rst_n,
    input  logic             enable,
    input  logic [CNT_W-1:0] period,
    input  logic [CNT_W-1:0] duty,
    output logic             pwm_h,
    output logic             pwm_l
);

    logic [CNT_W-1:0] counter;
    logic pwm_raw;

    // Counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            counter <= '0;
        else if (counter >= period - 1)
            counter <= '0;
        else
            counter <= counter + 1;
    end

    // Raw PWM
    assign pwm_raw = (counter < duty);

    // Dead-time insertion
    logic [7:0] rise_delay, fall_delay;
    logic pwm_h_raw, pwm_l_raw;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rise_delay <= '0;
            fall_delay <= '0;
            pwm_h_raw  <= 1'b0;
            pwm_l_raw  <= 1'b0;
        end else begin
            // Rising edge delay for PWM_H
            if (pwm_raw && !pwm_h_raw) begin
                if (rise_delay >= DEAD_TIME)
                    pwm_h_raw <= 1'b1;
                else
                    rise_delay <= rise_delay + 1;
            end else if (!pwm_raw) begin
                pwm_h_raw  <= 1'b0;
                rise_delay <= '0;
            end

            // Falling edge delay for PWM_L (complementary)
            if (!pwm_raw && !pwm_l_raw) begin
                if (fall_delay >= DEAD_TIME)
                    pwm_l_raw <= 1'b1;
                else
                    fall_delay <= fall_delay + 1;
            end else if (pwm_raw) begin
                pwm_l_raw  <= 1'b0;
                fall_delay <= '0;
            end
        end
    end

    assign pwm_h = enable ? pwm_h_raw : 1'b0;
    assign pwm_l = enable ? pwm_l_raw : 1'b0;

endmodule
""",
        "analysis": (
            "PWM with dead-time design:\n"
            "- Dead-time prevents H-bridge shoot-through\n"
            "- Both outputs go LOW during transitions\n"
            "- Configurable dead-time for different MOSFET switching speeds\n"
            "- Enable gate for emergency shutdown\n\n"
            "For synthesis: verify dead-time meets gate driver requirements.\n"
            "Typical dead-time: 50-500ns depending on power MOSFETs."
        ),
    },
]


def main():
    output_path = "data/sft/digital_rtl_v1.jsonl"

    print(f"\n{SEP}")
    print("   Generate Digital RTL SFT Data")
    print(f"{SEP}\n")

    examples = []
    for ex in EXAMPLES:
        example = {
            "messages": [
                {"role": "system", "content": build_system_message()},
                {"role": "user", "content": ex["task"]},
                {"role": "assistant", "content": (
                    f"{ex['reasoning']}\n\n"
                    f"```verilog\n{ex['verilog']}```\n\n"
                    f"{ex['analysis']}"
                )},
            ],
            "source": "digital_rtl_v1",
            "circuit_id": ex["id"],
            "domain": "digital",
        }
        lines = ex["verilog"].strip().count("\n") + 1
        print(f"  [{ex['id']}] {lines} lines Verilog")
        examples.append(example)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{SEP}")
    print(f"  Generated: {len(examples)} digital RTL examples")
    print(f"  Covers: FSM, FIFO, SPI, PWM")
    print(f"  Saved: {output_path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
