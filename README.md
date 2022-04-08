# OSNMA_ublox
Real-time authentication of ublox GNSS data using OSNMA and python

For MSc thesis GNSS Safety and Handling

Tested with Ublox ZED-F9P

Will write the following configurations to RAM 
• 42, CFG-NMEA-PROTVER
• 1, CFG-ITFM-ENABLE
• 2, CFG-ITFM-ANTSETTING
• 1, CFG-MSGOUT-NMEA ID GBS USB
• 1, CFG-MSGOUT-UBX MON RF USB
• 1, CFG-MSGOUT-UBX NAV CLOCK USB
• 1, CFG-MSGOUT-UBX NAV PVT USB
• 1, CFG-MSGOUT-UBX NAV SIG USB
• 1, CFG-MSGOUT-UBX NAV TIMEGAL USB
• 1, CFG-MSGOUT-UBX NAV STATUS USB
• 1, CFG-MSGOUT-UBX RXM RAWX USB
• 1, CFG-MSGOUT-UBX RXM SFRBX USB
• 0, CFG-SIGNAL-GLO ENA
• 0, CFG-SIGNAL-QZSS ENA
• 0, CFG-SIGNAL-BDS ENA

Monitors following messages:
['<NMEA(GAGSV', '<NMEA(GNGBS', ['<NMEA(GNGGA','<NMEA(GAGGA','<NMEA(GPGGA'],
 '<NMEA(GNGLL', '<NMEA(GNVTG', '<NMEA(GPGSV',
 '<UBX(MON-RF', '<UBX(MON-SPAN', '<UBX(NAV-CLOCK',
 '<UBX(NAV-PVT', '<UBX(NAV-SIG', '<UBX(NAV-TIMEGAL',
 '<UBX(RXM-RAWX', '<UBX(NAV-STATUS', '<UBX(RXM-SFRBX']
 
 Load from CSV file or record live data, Log to CSV file
 
Able to plot and monitor
 
Automatic Gain Control (AGC) from - MON-RF
(Average) Carrier to noise (CNO) from - RXM-RAWX
Pseudorange and doppler measurement consistency from - RXM-RAWX
RAIM status from - NMEA-Standard-GBS
Position and velocity consistency from - NAV-PVT
Time from: rcvToW from RXM-RAWX; galTow (Gallileo) and iTOW (gps) from UBX-NAV-TIMEGAL; iTOW, other estimates from NAV-CLOCK
 
Spectral diagram of U-blox Spetrum analyzer

Satellite Measurements (RXM - RAWX)

SFRBX DATA - OSNMA
  - Convert page bits according to Galileo ICD
  - Split pages by satellite PRN
  - Filter satellite which are actively transmitting OSNMA
  - Locate and store DSM-KROTO and TESLA root key, digital signatures
  - Assemble hash message and authenticate using ECDSA
  - Authenticate chain root key
  - Authenticate navigation tags
  - Read Galileo data words

Test OSNMA performance

TTFAF benchmarks

Time Mangement with NTP request



