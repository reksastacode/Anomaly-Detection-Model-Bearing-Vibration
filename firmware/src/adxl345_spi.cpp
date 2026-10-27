/*
 * ZENTRA TinyML - ADXL345 High-Speed SPI Driver Implementation (Tahap 11)
 * Target: ESP32-S3-DevKitC-1 N16R8
 */

#include <Arduino.h>
#include <SPI.h>
#include "adxl345_spi.h"
#include "zentra_inference.h"

// SPI settings: 2.0 MHz SPI clock, MSB first, SPI Mode 3 (CPOL=1, CPHA=1)
static const SPISettings ADXL_SPI_SETTINGS(2000000, MSBFIRST, SPI_MODE3);
static bool s_spi_initialized = false;

// Low-level SPI single register write
static void adxl345_write_reg(uint8_t cs_pin, uint8_t reg, uint8_t val) {
    SPI.beginTransaction(ADXL_SPI_SETTINGS);
    digitalWrite(cs_pin, LOW);
    SPI.transfer(reg & 0x3F); // Bit 7=0 (Write), Bit 6=0 (Single)
    SPI.transfer(val);
    digitalWrite(cs_pin, HIGH);
    SPI.endTransaction();
}

// Low-level SPI single register read
static uint8_t adxl345_read_reg(uint8_t cs_pin, uint8_t reg) {
    SPI.beginTransaction(ADXL_SPI_SETTINGS);
    digitalWrite(cs_pin, LOW);
    SPI.transfer(0x80 | (reg & 0x3F)); // Bit 7=1 (Read), Bit 6=0 (Single)
    uint8_t val = SPI.transfer(0x00);
    digitalWrite(cs_pin, HIGH);
    SPI.endTransaction();
    return val;
}

// Low-level SPI multi-byte burst read
static void adxl345_read_bytes(uint8_t cs_pin, uint8_t reg, uint8_t* dest, uint8_t len) {
    SPI.beginTransaction(ADXL_SPI_SETTINGS);
    digitalWrite(cs_pin, LOW);
    SPI.transfer(0xC0 | (reg & 0x3F)); // Bit 7=1 (Read), Bit 6=1 (Multi)
    for (uint8_t i = 0; i < len; i++) {
        dest[i] = SPI.transfer(0x00);
    }
    digitalWrite(cs_pin, HIGH);
    SPI.endTransaction();
}

bool adxl345_init(uint8_t cs_pin, uint8_t odr_code) {
    pinMode(cs_pin, OUTPUT);
    digitalWrite(cs_pin, HIGH);
    
    if (!s_spi_initialized) {
        // Initialize SPI on custom safe pins (SCK: 12, MISO: 13, MOSI: 11)
        SPI.begin(PIN_SPI_SCK, PIN_SPI_MISO, PIN_SPI_MOSI);
        s_spi_initialized = true;
    }
    
    delay(10);
    
    // 1. Verify Device ID
    uint8_t dev_id = adxl345_read_reg(cs_pin, ADXL345_REG_DEVID);
    Serial.printf("[ADXL345] Read DEVID on CS Pin %d: 0x%02X (Expected: 0x%02X)\n", 
                  cs_pin, dev_id, ADXL345_EXPECTED_DEVID);
    
    if (dev_id != ADXL345_EXPECTED_DEVID) {
        Serial.printf("[ADXL345 ERROR] Hardware ID mismatch on CS %d! Check wiring.\n", cs_pin);
        return false;
    }
    
    // 2. Put sensor in Standby while configuring
    adxl345_write_reg(cs_pin, ADXL345_REG_POWER_CTL, 0x00);
    
    // 3. Set Output Data Rate (ODR)
    adxl345_write_reg(cs_pin, ADXL345_REG_BW_RATE, odr_code & 0x0F);
    
    // 4. Set DATA_FORMAT: Full-Resolution mode, +/-16g range, 4-Wire SPI (Bit 3=1, Bit 1:0=11)
    adxl345_write_reg(cs_pin, ADXL345_REG_DATA_FORMAT, 0x0B);
    
    // 5. Configure FIFO: Stream Mode (replaces oldest data when full)
    adxl345_write_reg(cs_pin, ADXL345_REG_FIFO_CTL, 0x80 | 20); // Stream mode, 20-sample watermark
    
    // 6. Enter Measurement Mode
    adxl345_write_reg(cs_pin, ADXL345_REG_POWER_CTL, 0x08);
    
    Serial.printf("[ADXL345] Successfully initialized sensor on CS %d (ODR code: 0x%02X).\n", 
                  cs_pin, odr_code);
    return true;
}

bool adxl345_read_xyz(uint8_t cs_pin, float* x_g, float* y_g, float* z_g) {
    uint8_t raw_buf[6];
    adxl345_read_bytes(cs_pin, ADXL345_REG_DATAX0, raw_buf, 6);
    
    int16_t x_raw = (int16_t)(((uint16_t)raw_buf[1] << 8) | raw_buf[0]);
    int16_t y_raw = (int16_t)(((uint16_t)raw_buf[3] << 8) | raw_buf[2]);
    int16_t z_raw = (int16_t)(((uint16_t)raw_buf[5] << 8) | raw_buf[4]);
    
    *x_g = (float)x_raw * ADXL345_SCALE_FACTOR_G;
    *y_g = (float)y_raw * ADXL345_SCALE_FACTOR_G;
    *z_g = (float)z_raw * ADXL345_SCALE_FACTOR_G;
    
    return true;
}

uint8_t adxl345_read_fifo(uint8_t cs_pin, float* x_buf, float* y_buf, float* z_buf, uint8_t max_samples) {
    uint8_t fifo_status = adxl345_read_reg(cs_pin, ADXL345_REG_FIFO_STATUS);
    uint8_t entries = fifo_status & 0x3F; // Bits [5:0] indicate number of entries in FIFO (0-32)
    
    uint8_t to_read = (entries < max_samples) ? entries : max_samples;
    for (uint8_t i = 0; i < to_read; i++) {
        adxl345_read_xyz(cs_pin, &x_buf[i], &y_buf[i], &z_buf[i]);
    }
    return to_read;
}
