# MNIST V2: exact RV32IM inference, sparse pixels and eight register accumulators.
# Run: java -jar tools/rars.jar nc me ae2 se3 sm asm/mnist.s pa MODEL INPUT
# INPUT is either raw consecutive 784-byte images or '-' for persistent stdin.
# Persistent protocol: READY, then 1568 lowercase hex characters per line; QUIT.
# Neural-network math is identical to V1; only W1's storage order is transposed.
# h[j] = clip((B1[j] + sum(x[i] * W1[i,j])) >> shift, 0, 255)
# z[k] = B2[k] + sum(h[j] * W2[k,j]); first index wins ties.
# Pixel/activation loads are unsigned; weight loads and accumulators are signed.
# All-zero, sparse, and dense inputs follow the same exact arithmetic.
#
# Header: 8 little-endian words [0x324e4e52,2,784,64,10,shift,51112,0].
# Payload: W1 (784,64) i8; B1 offset50176 i32; W2 offset50432 (10,64) i8;
# B2 offset51072 i32. No weights are pruned and no arithmetic is approximated.

.data
.align 2
model_header: .space 32
model_data: .space 51112
image_data: .space 784
hidden_data: .space 64
scores_data: .space 40
# Each sparse record is (pixel_index * 64, unsigned pixel value), two words.
sparse_data: .space 6272
hex_data: .space 1571
extra_byte: .space 1

pred_prefix: .asciz "PRED "
ready_message: .asciz "READY\n"
err_args: .asciz "ERROR usage: mnist.s MODEL INPUT (two file paths required)\n"
err_model_open: .asciz "ERROR cannot open model file\n"
err_model_read: .asciz "ERROR model read failed or model is truncated\n"
err_model_header: .asciz "ERROR invalid model header or unsupported dimensions\n"
err_model_extra: .asciz "ERROR model contains unexpected trailing bytes\n"
err_input_open: .asciz "ERROR cannot open input file\n"
err_input_read: .asciz "ERROR input file read failed\n"
err_input_empty: .asciz "ERROR input file is empty\n"
err_input_partial: .asciz "ERROR input ends with a truncated 784-byte image\n"
err_hex: .asciz "ERROR malformed hex frame (expected 1568 lowercase hex characters)\n"

.text
.globl main
main:
    andi sp, sp, -16       # RARS initial sp is only word aligned.
    mv s0, a1
    li s1, -1             # model descriptor
    li s2, -1             # input descriptor
    li s3, 0              # number of processed images
    li s7, 0              # 0: raw file; 1: persistent hex stdin
    li t0, 2
    bne a0, t0, bad_args

    lw a0, 0(s0)
    li a1, 0
    li a2, 0
    li a7, 1024
    ecall
    bltz a0, bad_model_open
    mv s1, a0

    mv a0, s1
    la a1, model_header
    li a2, 32
    jal ra, read_full
    li t0, 32
    bne a0, t0, bad_model_read

    la t0, model_header
    lw t1, 0(t0)
    li t2, 0x324e4e52
    bne t1, t2, bad_model_header
    lw t1, 4(t0)
    li t2, 2
    bne t1, t2, bad_model_header
    lw t1, 8(t0)
    li t2, 784
    bne t1, t2, bad_model_header
    lw t1, 12(t0)
    li t2, 64
    bne t1, t2, bad_model_header
    lw t1, 16(t0)
    li t2, 10
    bne t1, t2, bad_model_header
    lw t1, 20(t0)
    li t2, 1
    blt t1, t2, bad_model_header
    li t2, 30
    bgt t1, t2, bad_model_header
    lw t1, 24(t0)
    li t2, 51112
    bne t1, t2, bad_model_header
    lw t1, 28(t0)
    bnez t1, bad_model_header

    mv a0, s1
    la a1, model_data
    li a2, 51112
    jal ra, read_full
    li t0, 51112
    bne a0, t0, bad_model_read
    mv a0, s1
    la a1, extra_byte
    li a2, 1
    jal ra, read_full
    bltz a0, bad_model_read
    bnez a0, bad_model_extra
    mv a0, s1
    li a7, 57
    ecall
    li s1, -1

    lw t0, 4(s0)
    lbu t1, 0(t0)
    li t2, 45              # '-' must be the complete argument.
    bne t1, t2, open_input
    lbu t1, 1(t0)
    bnez t1, open_input
    li s7, 1
    la a0, ready_message
    li a7, 4
    ecall
    j next_hex

open_input:
    mv a0, t0
    li a1, 0
    li a2, 0
    li a7, 1024
    ecall
    bltz a0, bad_input_open
    mv s2, a0

next_image:
    mv a0, s2
    la a1, image_data
    li a2, 784
    jal ra, read_full
    bltz a0, bad_input_read
    beqz a0, input_eof
    li t0, 784
    bne a0, t0, bad_input_partial
    j process_image

next_hex:
    # RARS ReadString reads a line through a BufferedReader, then appends '\n'
    # and NUL. An extra buffer character detects overlong frames after the
    # first 1568 characters. Its EOF result equals an empty line; both fail
    # immediately with ERROR rather than looping. QUIT is the clean shutdown.
    # A final line without a physical newline is normalized by RARS as well.
    la a0, hex_data
    li a1, 1571
    li a7, 8
    ecall
    la t0, hex_data
    lbu t1, 0(t0)
    li t2, 81              # Q
    bne t1, t2, decode_hex
    lbu t1, 1(t0)
    li t2, 85              # U
    bne t1, t2, bad_hex
    lbu t1, 2(t0)
    li t2, 73              # I
    bne t1, t2, bad_hex
    lbu t1, 3(t0)
    li t2, 84              # T
    bne t1, t2, bad_hex
    lbu t1, 4(t0)
    li t2, 10
    bne t1, t2, bad_hex
    lbu t1, 5(t0)
    bnez t1, bad_hex
    j successful_exit

decode_hex:
    la t1, image_data
    li t6, 784
    li a1, 9
    li a2, 5
decode_hex_pixel:
    lbu t2, 0(t0)
    addi t3, t2, -48
    bleu t3, a1, hex_high_ready
    addi t3, t2, -97
    bgtu t3, a2, bad_hex
    addi t3, t3, 10
hex_high_ready:
    slli t3, t3, 4
    lbu t2, 1(t0)
    addi t4, t2, -48
    bleu t4, a1, hex_low_ready
    addi t4, t2, -97
    bgtu t4, a2, bad_hex
    addi t4, t4, 10
hex_low_ready:
    or t3, t3, t4
    sb t3, 0(t1)
    addi t0, t0, 2
    addi t1, t1, 1
    addi t6, t6, -1
    bnez t6, decode_hex_pixel
    lbu t1, 0(t0)
    li t2, 10
    bne t1, t2, bad_hex
    lbu t1, 1(t0)
    bnez t1, bad_hex

process_image:
    la a0, image_data
    la a1, model_data
    la t0, model_header
    lw a2, 20(t0)
    la a3, hidden_data
    la a4, scores_data
    jal ra, infer
    mv s4, a0
    la a0, pred_prefix
    li a7, 4
    ecall
    mv a0, s4
    li a7, 1
    ecall
    la s5, scores_data
    li s6, 10
print_scores:
    li a0, 32
    li a7, 11
    ecall
    lw a0, 0(s5)
    li a7, 1
    ecall
    addi s5, s5, 4
    addi s6, s6, -1
    bnez s6, print_scores
    li a0, 10
    li a7, 11
    ecall
    addi s3, s3, 1
    bnez s7, next_hex
    j next_image

input_eof:
    beqz s3, bad_input_empty
    mv a0, s2
    li a7, 57
    ecall
    li s2, -1
successful_exit:
    li a0, 0
    li a7, 93
    ecall

bad_args:
    la a0, err_args
    j fatal
bad_model_open:
    la a0, err_model_open
    j fatal
bad_model_read:
    la a0, err_model_read
    j fatal
bad_model_header:
    la a0, err_model_header
    j fatal
bad_model_extra:
    la a0, err_model_extra
    j fatal
bad_input_open:
    la a0, err_input_open
    j fatal
bad_input_read:
    la a0, err_input_read
    j fatal
bad_input_empty:
    la a0, err_input_empty
    j fatal
bad_input_partial:
    la a0, err_input_partial
    j fatal
bad_hex:
    la a0, err_hex
    j fatal

fatal:
    mv s4, a0
    bltz s1, fatal_close_input
    mv a0, s1
    li a7, 57
    ecall
fatal_close_input:
    bltz s2, fatal_print
    mv a0, s2
    li a7, 57
    ecall
fatal_print:
    mv a0, s4
    li a7, 4
    ecall
    li a0, 1
    li a7, 93
    ecall

# read_full(fd, buffer, count) -> count read, 0 on EOF, -1 on read failure.
# Retries short reads and changes caller-saved registers only.
read_full:
    mv t0, a0
    mv t1, a1
    mv t2, a2
    li t3, 0
read_full_loop:
    beqz t2, read_full_done
    mv a0, t0
    mv a1, t1
    mv a2, t2
    li a7, 63
    ecall
    bltz a0, read_full_error
    beqz a0, read_full_done
    add t3, t3, a0
    add t1, t1, a0
    sub t2, t2, a0
    j read_full_loop
read_full_done:
    mv a0, t3
    ret
read_full_error:
    li a0, -1
    ret

# infer(pixels, payload, shift, hidden, scores) -> first maximum class.
# Owns a 64-byte aligned ABI frame; saves all twelve s registers and ra.
# First layer computes eight neurons simultaneously entirely in registers.
infer:
    addi sp, sp, -64
    sw ra, 60(sp)
    sw s0, 56(sp)
    sw s1, 52(sp)
    sw s2, 48(sp)
    sw s3, 44(sp)
    sw s4, 40(sp)
    sw s5, 36(sp)
    sw s6, 32(sp)
    sw s7, 28(sp)
    sw s8, 24(sp)
    sw s9, 20(sp)
    sw s10, 16(sp)
    sw s11, 12(sp)
    sw a3, 0(sp)
    sw a4, 4(sp)
    mv s0, a1              # current eight-neuron W1 column block
    la s1, sparse_data
    li t0, 50176
    add s3, a1, t0         # current eight-neuron B1 block
    mv s4, a3
    mv s5, a2

    # Scan once: skip only zero pixels. Store the W1 byte offset once so it
    # is reused in all eight blocks, with no multiply in the inner loop.
    mv t0, a0
    li t1, 784
    li t2, 0
    mv t3, s1
sparse_scan:
    lbu t4, 0(t0)
    beqz t4, sparse_scan_next
    sw t2, 0(t3)
    sw t4, 4(t3)
    addi t3, t3, 8
sparse_scan_next:
    addi t0, t0, 1
    addi t2, t2, 64
    addi t1, t1, -1
    bnez t1, sparse_scan
    mv s2, t3              # end pointer, including an empty sparse list
    li t5, 8               # 64 neurons / 8 accumulators
    li t6, 255

hidden_block:
    lw s6, 0(s3)
    lw s7, 4(s3)
    lw s8, 8(s3)
    lw s9, 12(s3)
    lw s10, 16(s3)
    lw s11, 20(s3)
    lw a0, 24(s3)
    lw a1, 28(s3)
    mv t2, s1
    beq t2, s2, hidden_block_done
hidden_dot8:
    lw t0, 0(t2)           # offset for this pixel's 64 contiguous weights
    lw t1, 4(t2)           # this positive pixel feeds all eight neurons
    add t0, s0, t0
    lb t3, 0(t0)
    mul t3, t3, t1
    add s6, s6, t3
    lb t3, 1(t0)
    mul t3, t3, t1
    add s7, s7, t3
    lb t3, 2(t0)
    mul t3, t3, t1
    add s8, s8, t3
    lb t3, 3(t0)
    mul t3, t3, t1
    add s9, s9, t3
    lb t3, 4(t0)
    mul t3, t3, t1
    add s10, s10, t3
    lb t3, 5(t0)
    mul t3, t3, t1
    add s11, s11, t3
    lb t3, 6(t0)
    mul t3, t3, t1
    add a0, a0, t3
    lb t3, 7(t0)
    mul t3, t3, t1
    add a1, a1, t3
    addi t2, t2, 8
    bltu t2, s2, hidden_dot8

hidden_block_done:
    sra s6, s6, s5
    bgez s6, clip0_positive
    li s6, 0
    j clip0_store
clip0_positive:
    ble s6, t6, clip0_store
    mv s6, t6
clip0_store:
    sb s6, 0(s4)
    sra s7, s7, s5
    bgez s7, clip1_positive
    li s7, 0
    j clip1_store
clip1_positive:
    ble s7, t6, clip1_store
    mv s7, t6
clip1_store:
    sb s7, 1(s4)
    sra s8, s8, s5
    bgez s8, clip2_positive
    li s8, 0
    j clip2_store
clip2_positive:
    ble s8, t6, clip2_store
    mv s8, t6
clip2_store:
    sb s8, 2(s4)
    sra s9, s9, s5
    bgez s9, clip3_positive
    li s9, 0
    j clip3_store
clip3_positive:
    ble s9, t6, clip3_store
    mv s9, t6
clip3_store:
    sb s9, 3(s4)
    sra s10, s10, s5
    bgez s10, clip4_positive
    li s10, 0
    j clip4_store
clip4_positive:
    ble s10, t6, clip4_store
    mv s10, t6
clip4_store:
    sb s10, 4(s4)
    sra s11, s11, s5
    bgez s11, clip5_positive
    li s11, 0
    j clip5_store
clip5_positive:
    ble s11, t6, clip5_store
    mv s11, t6
clip5_store:
    sb s11, 5(s4)
    sra a0, a0, s5
    bgez a0, clip6_positive
    li a0, 0
    j clip6_store
clip6_positive:
    ble a0, t6, clip6_store
    mv a0, t6
clip6_store:
    sb a0, 6(s4)
    sra a1, a1, s5
    bgez a1, clip7_positive
    li a1, 0
    j clip7_store
clip7_positive:
    ble a1, t6, clip7_store
    mv a1, t6
clip7_store:
    sb a1, 7(s4)
    addi s0, s0, 8
    addi s3, s3, 32
    addi s4, s4, 8
    addi t5, t5, -1
    bnez t5, hidden_block

    addi s0, s0, -64       # original payload base
    lw a0, 0(sp)
    li t0, 50432
    add a1, s0, t0
    li t0, 51072
    add a2, s0, t0
    lw a3, 4(sp)
    jal ra, dense_logits4
    lw a0, 4(sp)
    li a1, 10
    jal ra, argmax

    lw s11, 12(sp)
    lw s10, 16(sp)
    lw s9, 20(sp)
    lw s8, 24(sp)
    lw s7, 28(sp)
    lw s6, 32(sp)
    lw s5, 36(sp)
    lw s4, 40(sp)
    lw s3, 44(sp)
    lw s2, 48(sp)
    lw s1, 52(sp)
    lw s0, 56(sp)
    lw ra, 60(sp)
    addi sp, sp, 64
    ret

# dense_logits4(hidden, W2, B2, scores): ten rows of exactly 64 activations.
# Unroll four multiply-adds to cut per-element pointer and branch overhead.
dense_logits4:
    li a5, 10
logits_row:
    lw t0, 0(a2)
    mv t1, a0
    li t2, 16
logits_dot4:
    lbu t3, 0(t1)
    lb t4, 0(a1)
    mul t3, t3, t4
    add t0, t0, t3
    lbu t3, 1(t1)
    lb t4, 1(a1)
    mul t3, t3, t4
    add t0, t0, t3
    lbu t3, 2(t1)
    lb t4, 2(a1)
    mul t3, t3, t4
    add t0, t0, t3
    lbu t3, 3(t1)
    lb t4, 3(a1)
    mul t3, t3, t4
    add t0, t0, t3
    addi t1, t1, 4
    addi a1, a1, 4
    addi t2, t2, -1
    bnez t2, logits_dot4
    sw t0, 0(a3)
    addi a2, a2, 4
    addi a3, a3, 4
    addi a5, a5, -1
    bnez a5, logits_row
    ret

# Signed first-maximum argmax also works when all ten logits are negative.
argmax:
    mv t0, a0
    lw t3, 0(t0)
    li t1, 1
    li t2, 0
    addi t0, t0, 4
argmax_loop:
    bge t1, a1, argmax_done
    lw t4, 0(t0)
    ble t4, t3, argmax_next
    mv t3, t4
    mv t2, t1
argmax_next:
    addi t0, t0, 4
    addi t1, t1, 1
    j argmax_loop
argmax_done:
    mv a0, t2
    ret
