/*
 * ZENTRA TinyML - ADXL345 High-Speed SPI Driver Header (Tahap 11)
 * Target: ESP32-S3-DevKitC-1 N16R8
 * Supports: 4-Wire SPI @ 2-4 MHz, 3200 Hz ODR, FIFO Stream, Device ID 0xE5 verification.
 */

#ifndef ADXL345_SPI_H_
#define ADXL345_SPI_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================
// ADXL345 REGISTERS & CONTROL CODES
// ==============================================================================
#define ADXL345_REG_DEVID          0x00
#define ADXL345_REG_BW_RATE        0x2C
#define ADXL345_REG_POWER_CTL      0x2D
#define ADXL345_REG_INT_ENABLE     0x2E
#define ADXL345_REG_INT_MAP        0x2F
#define ADXL345_REG_INT_SOURCE     0x30
#define ADXL345_REG_DATA_FORMAT    0x31
#define ADXL345_REG_DATAX0         0x32
#define ADXL345_REG_FIFO_CTL       0x38
#define ADXL345_REG_FIFO_STATUS    0x39

#define ADXL345_EXPECTED_DEVID     0xE5

// ODR (Output Data Rate) Codes
#define ADXL345_ODR_3200HZ         0x0F  // Max ODR (Bandwidth ~1.6 kHz)
#define ADXL345_ODR_1600HZ         0x0E  // Bandwidth ~800 Hz
#define ADXL345_ODR_800HZ          0x0D
#define ADXL345_ODR_400HZ          0x0C

// Range & Scale Factor: Full-Resolution mode (+/-16g) is 3.9 mg/LSB (1/256 g)
#define ADXL345_SCALE_FACTOR_G     0.00390625f

// ==============================================================================
// DRIVER API PROTOTYPES
// ==============================================================================

/**
 * Initializes SPI bus and configures ADXL345 registers.
 * Verifies Device ID (must return 0xE5).
 * cs_pin: GPIO connected to CS pin (e.g., GPIO 10 or 9).
 * odr_code: e.g. ADXL345_ODR_3200HZ.
 */
bool adxl345_init(uint8_t cs_pin, uint8_t odr_code);

/**
 * Reads single XYZ acceleration sample in 'g' units.
 */
bool adxl345_read_xyz(uint8_t cs_pin, float* x_g, float* y_g, float* z_g);

/**
 * Reads burst data directly from ADXL345 hardware FIFO in Stream mode.
 * Returns the number of samples actually read (up to 32).
 */
uint8_t adxl345_read_fifo(uint8_t cs_pin, float* x_buf, float* y_buf, float* z_buf, uint8_t max_samples);

#ifdef __cplusplus
}
#endif

#endif // ADXL345_SPI_H_
